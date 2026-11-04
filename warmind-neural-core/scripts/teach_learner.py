"""End-to-end check: train the learner model only through RASPUTIN's
neural_core tool (trial -> reward), as an agent would."""
import json, os, random, sys, time
os.environ["NEURAL_CORE"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import neural_core as nc

n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
svc = nc.NeuralCoreService(mode="learner", autostart=False); svc.start()
while svc.status == "warming":
    time.sleep(1)
nc._service = svc
rng = random.Random(5)
log = []
t0 = time.time()
for k in range(n):
    cue = rng.choice("AB")
    out = nc.run_neural_core_tool({"action": "trial", "cue": cue})
    choice = int(out.split("action ")[1][0])
    correct = (choice == 1) == (cue == "A")
    nc.run_neural_core_tool({"action": "reward", "value": 1 if correct else -1, "note": "teacher"})
    log.append(int(correct))
    if (k + 1) % 20 == 0:
        print(f"trial {k + 1}: accuracy last 20 {sum(log[-20:]) / 20:.2f}  selectivity {svc.column.selectivity():.2f}  "
              f"wall {time.time() - t0:.0f}s", flush=True)
print(nc.run_neural_core_tool({"action": "status"}))
json.dump(dict(curve=log, selectivity=svc.column.selectivity()), open("teach_learner_result.json", "w"))
svc.stop()
