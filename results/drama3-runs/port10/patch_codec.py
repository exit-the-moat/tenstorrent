p = "host/stream_codec.py"
s = open(p).read()
old = """        w = next(w for w in self.windows if w >= n)
        z = self.dq(codes, w)
        audio = self.decs[w](z)
        if w not in self.dq.traces:
            ttnn.deallocate(z)
        return audio[keep_from * self.frame_length : n * self.frame_length]
"""
new = """        w = next(w for w in self.windows if w >= n)
        z = self.dq(codes, w)
        dt = getattr(self, "dev_traces", {}).get(w)
        if dt is not None:  # port10 DRAMA3_CODEC_DEV_TRACE=1: the decoder replayed on the quantizer trace's output
            tid, out, L = dt
            ttnn.execute_trace(self.dq.mesh, tid, cq_id=0, blocking=False)
            audio = ttnn.to_torch(out).float()[0, :L, 0]
            return audio[keep_from * self.frame_length : n * self.frame_length]
        audio = self.decs[w](z)
        if w not in self.dq.traces:
            ttnn.deallocate(z)
        return audio[keep_from * self.frame_length : n * self.frame_length]
"""
assert old in s
s = s.replace(old, new, 1)
old2 = """        if self.dq is not None:
            self.dq.capture()
"""
new2 = """        if self.dq is not None:
            self.dq.capture()
            if os.environ.get("DRAMA3_CODEC_DEV_TRACE", "0") == "1":
                # port10: each decoder as a trace whose input is the quantizer trace's persistent output z (same
                # programs as the eager decode_dev path, without the per-op host dispatch)
                import ttnn

                self.dev_traces = {}
                for w, d in self.decs.items():
                    if w not in self.dq.traces:
                        continue
                    z = self.dq.traces[w][2]
                    x, _ = d._body(z)
                    ttnn.deallocate(x)
                    ttnn.synchronize_device(self.dq.mesh)
                    tid = ttnn.begin_trace_capture(self.dq.mesh, cq_id=0)
                    out, _ = d._body(z)
                    ttnn.end_trace_capture(self.dq.mesh, tid, cq_id=0)
                    ttnn.synchronize_device(self.dq.mesh)
                    self.dev_traces[w] = (tid, out, d.L_out)
"""
assert old2 in s
s = s.replace(old2, new2, 1)
open(p, "w").write(s)
print("ok")
