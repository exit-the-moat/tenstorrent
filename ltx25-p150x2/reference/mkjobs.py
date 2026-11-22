"""Writes the job JSON files for the LTX-2.5 RTX PRO 6000 reference run (same prompts/seeds as the tt runs)."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
p = ("A lone cloaked figure stands on a dark stone castle wall at night holding a small glowing lantern and watching over "
     "a glowing teal moat below. The water ripples gently and softly reflects the light, faint mist drifts slowly over the "
     "moat, the lantern flame flickers. Very slow, subtle camera push-in. Painterly, cinematic, quiet ambient night sound, "
     "no speech.")
w = lambda n, o: json.dump(o, open(os.path.join(H, n), "w"), indent=1)
w("i2v5s.json", [{"name": "moatwatcher_i2v", "seed": 0, "prompt": p, "image": "Moatwatcher.png"}])
w("i2vlong.json", [{"name": "moatwatcher_single", "seed": 0, "prompt": p + " One single continuous shot, no cuts.",
                    "image": "Moatwatcher.png"}])
h = json.load(open(os.path.join(H, "harness8.json")))
w("motion1.json", [h[0]])
print("ok")
