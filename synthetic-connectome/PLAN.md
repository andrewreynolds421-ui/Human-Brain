# VEGA: research review and plan (27 November 2026)

Written at the user's request: stop and look at who else is building something like VEGA, what they found, and
what that says VEGA should do next. Sources are the published papers named below (found by web search on this
date); nothing here is a new result. Decisions are the user's; this file is the proposal.

## 1. Where VEGA is

- Cortex side: a reading network (next-word prediction). Stage 94: raised on connected text, a large recurrent
  reader uses the story so far and reads real-book passages at 42 % by itself (adults ~82 % on the published test).
- Hippocampus: sentence events, pattern completion, three recall steps, roles; taught recall steps. bAbI, one answer
  list: 12-14 of 20 tasks, 8.7 % error at 10x training (MemN2N 14 / 7.5 %).
- Sleep / replay prevents forgetting (Stage 86). Dopamine-style learning and hindsight: Stage 96, running.
- Missing: combining facts in space (left / right at chance), memory of real stories, any grounding of words, any
  transfer of what the hippocampus learns back into the cortex, one system doing all of it at once.

## 2. What others built, and the lesson for VEGA

| project | what it is | lesson |
|---|---|---|
| Spaun (Eliasmith et al. 2012, Science) | 2.5 million spiking neurons, one fixed circuit, eight tasks, sees images and moves an arm | One architecture for all tasks, compared with human behaviour (errors, timing). Its authors' view: cognition does not emerge from wiring data alone - function has to be designed or trained in. Matches our wired-hippocampus result. |
| PBWM / Leabra (O'Reilly & Frank 2006; Hazy et al. 2007) | prefrontal cortex holds working memory; basal ganglia gate what enters and leaves it; an actor-critic dopamine system trains the gate | The closest existing model to the user's "reward and punishment to keep focus". Gating learned by dopamine, representations learned by error correction - both, as the user proposed. VEGA's working memory is ungated. |
| Complementary learning systems (McClelland et al. 1995; Kumaran, Hassabis & McClelland 2016; Sun et al. 2023) | fast hippocampus + slow cortex; replay moves knowledge to cortex, weighted by goals; consolidate what generalises | VEGA replays only inside the hippocampus's own training. Nothing the hippocampus learns ever reaches the reading network. That loop is the theory's core and VEGA does not have it. |
| Tolman-Eichenbaum Machine (Whittington et al. 2020); clone-structured graphs (George et al. 2021; Raju et al. 2024) | hippocampus + entorhinal cortex learn the structure of a world (a latent graph / map) from sequences, reuse it in new worlds; transitive inference, shortcuts, planning | The "mental model" VEGA lacks. Structure is learned once from many small worlds and re-bound to new contents - not learned from question answers (our Stage 92 failure). |
| Entity Network (Henaff et al. 2017) | memory cells tracking each thing's state while reading; solves all 20 bAbI tasks | Cells tied to the things in the story, ~200 passes of training. Ours (Stage 93): free cells, 4 passes - not a fair copy yet. |
| Attention-Sum Reader / BookTest (Kadlec et al. 2016; Bajgar et al. 2016) | answer by pointing at a word in the passage; data mattered more than architecture | 10x more passage training, pointer answers. |
| BabyLM (Warstadt et al. 2023; Hu et al. 2024) | language models on <= 100 million words | Architecture, objective and compute all matter at child scale. Our reader has read 5-13 million words. |
| Child-view learning (Vong et al. 2024, Science) | a generic network learns word meanings from 61 hours of one child's head-camera video and speech | Grounding is learnable from small, real data. VEGA's words are tied to nothing but other words. (Data: SAYCam, by application.) |
| Thousand Brains / Monty (Clay et al. 2024; Leadholm et al. 2025) | cortical-column modules with reference frames, learning by moving sensors; fast local learning | Structure via reference frames; learning by acting. Early stage (object recognition). |
| LeCun 2022 (position paper) | perception, world model, cost + critic, short-term memory, actor, configurator | A world model that predicts what happens next in its own representation, used to plan. VEGA predicts words, not situations. |
| Connectome-constrained networks (Lappalainen et al. 2024, Nature) | fly visual system: measured wiring + training on the task predicts real neural activity | The credible way to use a connectome: wiring as constraint AND task training together. Wiring alone or training alone did worse. |

## 3. Theory: what VEGA should be

Five loops, each with a brain counterpart and a published working model:

1. **Read** (cortex): a story-raised reader, sized to carry a paragraph. Slow learning.
2. **Remember** (hippocampus): events, pattern completion. Fast learning. Have it.
3. **Hold and focus** (prefrontal cortex + basal ganglia): a small gated working memory; what to keep, what to
   recall next, when to answer - chosen, and trained by dopamine (reward prediction error); the contents trained by
   correction and hindsight. PBWM is the template. Stage 96 is the first step.
4. **Model the situation** (hippocampus + entorhinal cortex): a workspace where the things in a story are placed
   in a learned structure - a line, a map, a family tree - and new relations are read off. Structure learned from
   many generated worlds where the truth is known, then re-bound to new stories (TEM / clone graphs).
5. **Consolidate** (sleep): replay from hippocampus trains the cortex, so that what was an episode becomes
   knowledge and the next story is read with it. Goal-weighted (dopamine tags what to replay).

And two commitments from the review:
- **One VEGA, one battery.** Stop training a new network per stage. One system, tested on the same fixed set after
  every change: grade test, BLiMP, bAbI (one answer list), world stories, passages with baselines, and - new -
  transfer tests (a skill taught on one kind of problem, tested on another).
- **Constraint + training.** Where the connectome gives numbers, use them as constraints and train the function
  (the fly result); do not expect function from wiring alone.

## 4. Proposed order (each a few days; first number is the order)

1. **Finish what is running** (Stages 95, 96): big reader under the hippocampus; dopamine + hindsight.
2. **Gated working memory (PBWM-style)**: slots, input / output gates as rewarded choices, critic; replaces the
   soft controller. Test: bAbI without sentence labels vs with; steps taken per task; does it stay on task over
   long stories.
3. **Situation workspace**: pre-train on generated worlds (orderings, 2-D layouts, room graphs) with known truth;
   then bAbI 17 / 19 and deduction as transfer tests, never trained on directly at first.
4. **Consolidation**: nights where hippocampal replay trains the reading network; test whether tomorrow's reading
   (passage test, new stories about the same things) improves, and that old reading is not lost.
5. **Scale the reader** on all books we have (and more if the user adds them), longer lessons, with the grade test
   and BLiMP as gates; fix and retest the attention reader.
6. **Real books**: pointer answers, 100,000 passages, hindsight.
7. Later, needing data from the user: grounding (child-view video, picture books), then research skills on top of
   a working focus-and-recall loop.

## 5. Open questions for the user

- One combined VEGA from now on (slower per experiment, but honest about the whole), or keep fast separate tests
  and merge monthly?
- How far to scale: GitHub runners give ~8 hours per large reader; a rented GPU would give 50-100x.
- Is grounding (vision) in scope soon? It needs data only the user can obtain.
