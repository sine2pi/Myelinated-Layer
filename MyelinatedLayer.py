

    class MyelinatedLayer(BaseAttention):
        def __init__(self, dims, head, layerAs=6, sparsity_threshold=0.1):
            super().__init__()
            self.layers = nn.ModuleList()
            self.layerAs = layerAs
            self.sparsity_threshold = sparsity_threshold
            
            self.shared_head = AdaptiveSpan(dims, head)
            
            self.node_predictors = nn.ModuleList([
                nn.Sequential(
                    LayerNorm(dims),
                    Linear(dims, 1),
                    nn.Sigmoid()
                ) for _ in range(layerAs)
            ])
            
            for i in range(layerAs):
                self.layers.append(nn.ModuleDict({
                    'ln': LayerNorm(dims),
                    'gate': nn.Sequential(Linear(dims, 1), nn.Sigmoid()),
                    'adapter': Linear(dims, dims) if i % 2 == 0 else None
                }))
            
            self.policy_net = nn.Sequential(
                Linear(dims, 128),
                nn.ReLU(),
                Linear(128, 3)
            )
            
            self.jump_weights = nn.Parameter(torch.tensor([0.1, 0.05, 0.01]))
            
            n_mlp = dims * 4
            self.mlp_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.mlp = nn.Sequential(Linear(dims, n_mlp), nn.GELU(), Linear(n_mlp, dims))
            self.mlp_ln = LayerNorm(dims)
            
            self.working_memory = nn.Parameter(torch.zeros(1, 1, dims))
            self.memory_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            
        def shared_head(self, norm_x, mask=None, kv_cache=None):
            batch_size, seq_len = norm_x.shape[:2]
            
            q = norm_x.view(batch_size, seq_len, self.head, -1).transpose(1, 2)
            k = norm_x.view(batch_size, seq_len, self.head, -1).transpose(1, 2)
            v = norm_x.view(batch_size, seq_len, self.head, -1).transpose(1, 2)
            
            attn_output, _ = calculate_attention(q, k, v, mask, 1.0, BaseAttention.use_sdpa)
            
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            return attn_output


        def predict_node_importance(self, x, layer_idx):
            """Dynamically determine if processing should occur at this node"""
            importance = self.node_predictors[layer_idx](x)
            return (importance > self.sparsity_threshold).float()
        
        def forward(self, x, xa=None, mask=None, kv_cache=None):
            batch_size, seq_len = x.shape[:2]
            
            working_memory = self.working_memory.expand(batch_size, -1, -1)
            
            original_x = x
            
            pooled_representation = x.mean(dim=1)
            policy_logits = self.policy_net(pooled_representation)
            policy = F.softmax(policy_logits, dim=-1)
            
            jump_history = []
            i = 0
            while i < self.layerAs:
                layer = self.layers[i]
                
                node_importance = self.predict_node_importance(x, i)
                
                if node_importance.mean() < 0.2 and i > 0:
                    i += 1
                    jump_history.append(i)
                    continue
                    
                norm_x = layer['ln'](x)
                
                attn_mask = mask
                if mask is None:
                    attn_mask = node_importance.squeeze(-1).unsqueeze(1).expand(-1, seq_len, -1)
                else:
                    attn_mask = mask * node_importance.squeeze(-1).unsqueeze(1).expand(-1, seq_len, -1)
                    
                if node_importance.mean() > 0.3:
                    attn_output = self.shared_head(norm_x, mask=attn_mask, kv_cache=kv_cache)[0]
                    
                    if layer['adapter'] is not None:
                        attn_output = layer['adapter'](attn_output)
                    
                    gate_value = layer['gate'](norm_x).unsqueeze(-1)
                    x = x + gate_value * attn_output
                    
                    memory_gate = self.memory_gate(x)
                    working_memory = memory_gate * working_memory + (1 - memory_gate) * x.mean(dim=1, keepdim=True)
                
                jump_prob = policy[:, 1] if i < self.layerAs - 1 else torch.zeros_like(policy[:, 1])
                should_jump = (torch.rand_like(jump_prob) < jump_prob).any()
                
                if should_jump:
                    jump_length = torch.multinomial(policy, 1)[:, 0].max().item() + 1
                    
                    i_next = min(i + jump_length, self.layerAs - 1)
                    skip_weight = self.jump_weights[min(jump_length-1, 2)]
                    
                    x = x + skip_weight * original_x + (1-skip_weight) * working_memory
                    
                    i = i_next
                    jump_history.append(i)
                else:
                    i += 1
            
            mlp_importance = self.mlp_gate(x)
            mlp_output = self.mlp(self.mlp_ln(x))
            x = x + mlp_importance * mlp_output
            
            return x, {'jumps': jump_history}

class MyelinatedLayer(nn.Module):
    def __init__(self, dims, head, n_layers=6, sparsity_threshold=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.sparsity_threshold = sparsity_threshold
        
        # Create shared attention components with parameter-efficient design
        self.shared_head = nn.MultiheadAttention(dims, head)
        
        # Node importance predictors (analogous to Nodes of Ranvier density)
        self.node_predictors = nn.ModuleList([
            nn.Sequential(
                LayerNorm(dims),
                Linear(dims, 1),
                nn.Sigmoid()
            ) for _ in range(n_layers)
        ])
        
        # Create efficient gating networks for each processing node
        for i in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'ln': LayerNorm(dims),
                'gate': nn.Sequential(Linear(dims, 1), nn.Sigmoid()),
                'adapter': Linear(dims, dims) if i % 2 == 0 else None
            }))
        
        # Efficient neural policy network instead of table-based RL
        self.policy_net = nn.Sequential(
            Linear(dims, 128),
            nn.ReLU(),
            Linear(128, 3)  # 3 jumping patterns
        )
        
        # Trainable jumping weights
        self.jump_weights = nn.Parameter(torch.tensor([0.1, 0.05, 0.01]))
        
        # MLP with conditional computation
        n_mlp = dims * 4
        self.mlp_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        self.mlp = nn.Sequential(Linear(dims, n_mlp), nn.GELU(), Linear(n_mlp, dims))
        self.mlp_ln = LayerNorm(dims)
        
        # Memory systems 
        self.working_memory = nn.Parameter(torch.zeros(1, 1, dims))
        self.memory_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        
    def predict_node_importance(self, x, layer_idx):
        """Dynamically determine if processing should occur at this node"""
        # Token-level importance prediction
        importance = self.node_predictors[layer_idx](x)
        return (importance > self.sparsity_threshold).float()
    
    def forward(self, x, xa=None, mask=None, kv_cache=None):
        batch_size, seq_len = x.shape[:2]
        
        # Prepare memory
        working_memory = self.working_memory.expand(batch_size, -1, -1)
        
        # Original signal preservation
        original_x = x
        
        # Get adaptive policy from input content
        pooled_representation = x.mean(dim=1)
        policy_logits = self.policy_net(pooled_representation)
        policy = F.softmax(policy_logits, dim=-1)
        
        # Dynamic path through network
        jump_history = []
        i = 0
        while i < self.n_layers:
            layer = self.layers[i]
            
            # Predict importance for this layer
            node_importance = self.predict_node_importance(x, i)
            
            # Skip layers with low importance (sparse computation)
            if node_importance.mean() < 0.2 and i > 0:
                i += 1
                jump_history.append(i)
                continue
                
            # Process with attention only where important
            norm_x = layer['ln'](x)
            
            # Compute attention with token-level importance masking
            attn_mask = mask
            if mask is None:
                attn_mask = node_importance.squeeze(-1).unsqueeze(1).expand(-1, seq_len, -1)
            else:
                attn_mask = mask * node_importance.squeeze(-1).unsqueeze(1).expand(-1, seq_len, -1)
                
            # Compute attention only if this layer has sufficient importance
            if node_importance.mean() > 0.3:
                attn_output = self.shared_head(norm_x, mask=attn_mask, kv_cache=kv_cache)[0]
                
                # Apply layer-specific adaptation if available (specialized nodes)
                if layer['adapter'] is not None:
                    attn_output = layer['adapter'](attn_output)
                
                # Dynamic gating of attention influence
                gate_value = layer['gate'](norm_x).unsqueeze(-1)
                x = x + gate_value * attn_output
                
                # Integrate with working memory
                memory_gate = self.memory_gate(x)
                working_memory = memory_gate * working_memory + (1 - memory_gate) * x.mean(dim=1, keepdim=True)
            
            # Decide jumping based on policy
            jump_prob = policy[:, 1] if i < self.n_layers - 1 else torch.zeros_like(policy[:, 1])
            should_jump = (torch.rand_like(jump_prob) < jump_prob).any()
            
            if should_jump:
                # Jump length based on policy
                jump_length = torch.multinomial(policy, 1)[:, 0].max().item() + 1
                
                # Mix signals - weighted by learned jump weights
                i_next = min(i + jump_length, self.n_layers - 1)
                skip_weight = self.jump_weights[min(jump_length-1, 2)]
                
                # Skip connection with content-dependent mixing
                x = x + skip_weight * original_x + (1-skip_weight) * working_memory
                
                i = i_next
                jump_history.append(i)
            else:
                i += 1
        
        # Conditional MLP computation
        mlp_importance = self.mlp_gate(x)
        mlp_output = self.mlp(self.mlp_ln(x))
        x = x + mlp_importance * mlp_output
        
        # Add jumping behavior to output for analysis
        return x, {'jumps': jump_history}




class MyelinatedLayerb(nn.Module):
    def __init__(self, dims, head, n_layers=6, rl_states=256, rl_actions=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        
        # Create nodes of Ranvier and myelinated segments
        for i in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(dims, head),
                'ln': LayerNorm(dims),
                'is_node': torch.nn.Parameter(torch.tensor(i % 2 == 0), requires_grad=False)  # Initial node pattern
            }))
        
        # Reinforcement learning for adaptive jumping behavior
        self.refiner = Refiner(states=rl_states, actions=rl_actions, 
                              alpha=0.05, gamma=0.95, epsilon=0.15)
        
        # State extraction and quality assessment networks
        self.state_extractor = nn.Sequential(
            Linear(dims, dims // 4),
            nn.GELU(),
            Linear(dims // 4, 16)  # Compress to a manageable state representation
        )
        
        self.quality_assessor = nn.Sequential(
            Linear(dims, dims // 4),
            nn.GELU(),
            Linear(dims // 4, 1)
        )
        
        # MLP for terminal processing
        n_mlp = dims * 4
        self.mlp = nn.Sequential(Linear(dims, n_mlp), nn.GELU(), Linear(n_mlp, dims))
        self.mlp_ln = LayerNorm(dims)
    
    def extract_state(self, x):
        # Extract a compact state representation for RL
        pooled = x.mean(dim=1)  # Global pooling
        state_features = self.state_extractor(pooled)
        # Discretize for table-based RL
        state_id = (state_features.sigmoid() * 15).round().sum().int().item()
        return state_id
    
    def assess_quality(self, x):
        # Assess quality of processing (reward signal)
        pooled = x.mean(dim=1)
        quality = self.quality_assessor(pooled).mean().item()
        return quality
    
    def action_to_jump_pattern(self, action):
        """Convert RL action to jumping pattern"""
        if action == 0:
            return 1  # Standard sequential processing
        elif action == 1:
            return 2  # Skip every other node
        else:
            return 3  # Long jumps
    
    def forward(self, x, xa=None, mask=None, kv_cache=None):
        # Keep original signal for long-range jumps
        original_x = x
        last_node_x = x
        
        # Extract state and decide jumping pattern
        state = self.extract_state(x)
        action = self.refiner.choose_action(state)
        jump_pattern = self.action_to_jump_pattern(action)
        
        # Process through axon segments with adaptive jumping
        i = 0
        while i < len(self.layers):
            layer = self.layers[i]
            
            if layer['is_node']:
                # At nodes, perform full attention computation
                node_output = layer['attn'](layer['ln'](x), mask=mask, kv_cache=kv_cache)[0]
                x = x + node_output
                
                # Store node output for potential jumping
                last_node_x = x
                
                # Apply jumping based on learned pattern
                if jump_pattern > 1 and i > 0:
                    # Skip ahead based on jump pattern
                    i += jump_pattern
                    # Mix in original signal with adaptive weight based on position
                    skip_weight = 0.1 * (1.0 - i/len(self.layers))
                    x = x + original_x * skip_weight + last_node_x * (0.1 - skip_weight)
                    continue
            else:
                # In "myelinated" segments, signal passes with reduced computation
                x = x + 0.01 * layer['attn'](layer['ln'](x), mask=mask, kv_cache=kv_cache)[0]
            
            i += 1
        
        # Terminal processing
        x = x + self.mlp(self.mlp_ln(x))
        
        # Learn from results
        next_state = self.extract_state(x)
        quality = self.assess_quality(x)
        self.refiner.update(state, action, quality, next_state)
        
        return x
    
class MyelinatedLayerc(nn.Module):
    def __init__(self, dims, head, n_layers=4, jump_interval=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.jump_interval = jump_interval
        
        for i in range(n_layers):
            # Create each "node of Ranvier" (attention processing unit)
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(dims, head),
                'ln': LayerNorm(dims),
                'is_node': torch.tensor(i % jump_interval == 0)  # Mark nodes for processing
            }))
            
        # MLP at the end (terminal processing)
        n_mlp = dims * 4
        self.mlp = nn.Sequential(Linear(dims, n_mlp), nn.GELU(), Linear(n_mlp, dims))
        self.mlp_ln = LayerNorm(dims)
            
    def forward(self, x, xa=None, mask=None, kv_cache=None):
        # Keep original signal for long-range jumps
        original_x = x
        
        # Process through "axon segments" with nodes of Ranvier
        for i, layer in enumerate(self.layers):
            if layer['is_node']:
                # At nodes, perform full attention computation
                node_output = layer['attn'](layer['ln'](x), mask=mask, kv_cache=kv_cache)[0]
                
                # Add signal from most recent node AND from origin (long-range skip)
                x = x + node_output
                
                # Every jump_interval, mix in the original signal (saltatory conduction)
                if i > 0 and i % self.jump_interval == 0:
                    x = x + original_x * 0.1  # Scale factor for stability
            else:
                # In "myelinated" segments, signal passes with minimal processing
                # (still passes through but with reduced computation)
                x = x + 0.01 * layer['attn'](layer['ln'](x), mask=mask, kv_cache=kv_cache)[0]
        
        # Terminal processing
        x = x + self.mlp(self.mlp_ln(x))
        return x
    
