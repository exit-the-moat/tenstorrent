"""Day 4b: opt-in router shortcut (QWEN36_ROUTER_TOPK_FIRST=1): top-k on the raw scores, softmax over the k."""
p = "/opt/ep0/tt-metal/models/demos/blackhole/qwen36/tt/moe/router.py"
s = open(p).read()
old = '''        expert_scores = ttnn.linear(hidden_states, self.proj_weight, compute_kernel_config=self.compute_kernel_config)
        router_probs = ttnn.softmax(expert_scores, dim=-1)
        expert_scores.deallocate(True)
        top_k_values, top_k_indices = ttnn.topk(router_probs, k=self.top_k, dim=-1)
        router_probs.deallocate(True)
        if self.norm_topk_prob:'''
assert s.count(old) == 1
s = s.replace(old, '''        expert_scores = ttnn.linear(hidden_states, self.proj_weight, compute_kernel_config=self.compute_kernel_config)
        if self.norm_topk_prob and os.environ.get("QWEN36_ROUTER_TOPK_FIRST", "0") == "1":
            # softmax is monotonic and the renormalised top-k of softmax(s) is softmax(top-k of s): 3 ops fewer
            top_k_scores, top_k_indices = ttnn.topk(expert_scores, k=self.top_k, dim=-1)
            expert_scores.deallocate(True)
            top_k_values = ttnn.softmax(top_k_scores, dim=-1)
            top_k_scores.deallocate(True)
            return top_k_values, top_k_indices
        router_probs = ttnn.softmax(expert_scores, dim=-1)
        expert_scores.deallocate(True)
        top_k_values, top_k_indices = ttnn.topk(router_probs, k=self.top_k, dim=-1)
        router_probs.deallocate(True)
        if self.norm_topk_prob:''')
if "\nimport os\n" not in s:
    s = s.replace("import ttnn\n", "import os\n\nimport ttnn\n", 1)
open(p, "w").write(s)
print("ok")
