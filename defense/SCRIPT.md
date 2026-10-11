# Defense script · 20 min

Do not read the title. Speak in paragraphs. **Bold** = point there, then say the paragraph. Times are a ceiling.

---

## 1. Title · 0:20

Good morning. My name is Adrian Cebrian Urios.

This thesis takes the NN2Logic hybrid from a fully connected occupancy network to tiny quantised convolutional classifiers, and measures it on a simulated Ibex RISC-V core.

*click*

---

## 2. Outline · 0:20

Twenty minutes, four parts.

**01 Motivation**
Local inference on a microcontroller, why CNNs still pay the same MACs, the ReLU tree, and NN2Logic on a fully connected network.

**02 Approach**
The pipeline, how the CNN is rewritten for NN2Logic, and how first-stage routes become a hybrid program.

**03 Evaluation**
Four networks, a matched looped reference. Cycles on Spike, the ISA that Ibex implements.

**04 Findings**
Results, then the limits of an early exit.

*click*

---

## 3. Motivation · 0:05

Motivation.

*click*

---

## 4. TinyML setting · 0:45

TinyML here is the setting, not a network: a classifier that has to run next to the sensor, on a microcontroller, not in a data centre.

**Occupancy · Activity · Screening**
Occupancy in a building, activity from a wearable, medical screening at the point of care. The radio, the latency, and the payload keep the data on the device.

**THE DEVICE**
That device is a microcontroller. There is no GPU. The core is a general-purpose CPU — control flow and integer logic — here a small open RISC-V core, Ibex.

*click*

---

## 5. Tiny CNNs · 0:50

CNNs are the default model for the visual and short time-series tasks in this setting.

**Figure**
This is the usual block. Input, a convolution, average pooling, ReLU, then a small fully connected head. Almost all of the arithmetic is sliding-window MACs in those first stages.

The TinyML literature makes those MACs cheaper: integer QAT, pruning, architecture search, scheduled loops. Those methods explain why a small network can run on a microcontroller at all.

**The kind of computation stays the same**
Every sample still walks this block. That is a poor match for this CPU: most of its area is control flow and integer logic, and it has only a few multipliers.

*click*

---

## 6. ReLU is a decision tree · 0:45

The last slide was the same arithmetic for every sample. There is another way to look at the same network.

A ReLU does one thing: if the incoming value is positive, it keeps it; otherwise it is zero. That is a yes-or-no test on the sign.

**The network**
On the left is a tiny example: inputs, a few ReLUs, a class. The drawing is Stein et al., NN2Logic, 2026.

**Equals**
That network is this tree. Not a second model trained to copy it. With ReLU activations they compute the same function.

**The tree**
Each diamond is one ReLU: true or false, you take a branch. Follow the signs and you land on a class. Once the signs are known, that path is one linear piece of the network.

**Early exit, still MACs**
The usual way to skip work still leaves neural arithmetic: fewer layers, but still MACs.

**A path on this CPU**
A path through the tree is integer comparisons and only the MACs that path still needs. That mix is closer to what this microcontroller CPU already does.

*click*

---

## 7. NN2Logic · 1:10

NN2Logic, from TU Darmstadt, turns that tree into a program for a quantized integer network.

**Explore → Certify → Emit**
Explore walks the ReLU signs seen in training and keeps only the branches a sample actually took. Certify asks whether a leaf is one class on the whole input cell, not only on the samples that reached it. Emit writes those paths as C. A miss falls through to the original integer network. Both sides compute the same function. They differ only in how much of the network runs.

**Figure**
Left: the explored tree. Diamonds are tests, L0 to L4 are leaves. Right: the generated program. A certified prefix returns. A miss continues as the integer network. This figure is Stein et al., NN2Logic, 2026.

**Occupancy perceptron**
That hybrid was shown on an occupancy perceptron. Hidden layers of similar width, so a short prefix can skip a large share.

**Convolutional models**
Convolutional models were left open, and generated code size was not reported. That is the starting point of this thesis.

*click*

---

## 8. Where the MACs sit · 0:40

TinyML vision and sensing use convolutional models, not the small fully connected networks NN2Logic was shown on. On a CNN the cost sits here.

**Blue bar**
Almost all of the MACs are in the convolution and the pool. The head is a short vector. Skipping the head after a full convolution saves little.

**Test in the convolution**
The rest of the map is not computed. This is the saving we want.

**Test in the head**
The convolution has already run. Almost nothing is skipped.

*click*

---

## 9. Approach · 0:05

Approach.

*click*

---

## 10. Pipeline · 0:50

This is the workflow. Walk it left to right.

**Grey**
Float training, QAT, packaging, tree exploration, leaf certification. That is NN2Logic, called unchanged.

**Blue**
CNN export and linearisation, route selection, then the two code generators. Those stages are this thesis.

The emitter writes a hybrid and a matched reference from the same weights. Both go to the same RISC-V build.

A constant leaf the certifier never returned cannot be recovered later.

*click*

---

## 11. The quantized CNN · 0:50

NN2Logic only understands a list of dense integer layers with ReLUs in between. It has no idea what a convolution is. So the CNN is rewritten into that form. The tree builder itself does not change.

**1 Unfold**
A convolution slides a small window over the image. Unfolding writes that as a table: each output position is one row, and the pixels that window sees are the columns. After this, the layer looks like any other dense layer.

**2 Fuse**
Convolution, batch-norm, and average pooling become one integer layer. Average pooling is just a mean of neighbouring cells, so it can join that layer. Max pooling cannot: it picks a winner, it is not a sum. Dropout is only used in training, so it is not in the exported network.

**3 ReLU after pool**
The tree asks one yes-or-no per ReLU. If the ReLU sits after every convolution output, that is one test per pixel. After the pool, it is one test per pooled cell. The network is trained that way, so the tree sees the smaller set of tests.

**4 Recover geometry**
Unfolding forgets the kernel size, the stride, and the padding. The tree does not need them. The C emitter does, because it has to write a loop over the image. Those numbers are stored next to the tree.

**Blue bar**
The network is trained in 8-bit integers. That integer network is what NN2Logic sees, not the float model from the start of training.

*click*

---

## 12. Route selection · 0:30

Four steps, in this order. The next three slides unpack the first three.

**1 Trim** — shorten the path into the convolution.
**2 Vet** — may this route enter the pool?
**3 Score** — charge the work that still remains.
**4 Embed** — a short list, cheapest first.

The class of a route is the integer network’s own prediction, not the dataset label. A route is there to reproduce the fallback.

*click*

---

## 13. Path trimming · 0:45

The first step is the path trimming, which consists of reducing the size of branches, in order to make them simpler and cheaper. We start from the explored path to a constant leaf.

**t1–t6**
In this example we can see a path with six decisions, with three of them inside the convolutional layer, and three of them on the fully connected layer.

**Arrow / t1, t2**
By dropping the last four decisions, we only keep two that do not need to compute the whole convolution, making the path much cheaper. We keep the shortest prefix that still separates the leaf, then drop redundant tests.

This trimming just drops redundant tests on the training samples, so it is a safe trimming. That is how a test in the head becomes a test that can skip the map. I do this on the explored path. The published certificate has no redundant tests left to drop, and on these CNNs it still names a head test, so it is not the selector I report.

*click*

---

## 14. Vetting · 0:50

The second part is the vetting, which filters the total number of trimmed branches, and just keeps those that are statistically safe for embedding.

**1 No disagreements**
It rejects those branches which have at least one disagreement with the full integer network.

**2 Laplace support**
It applies a Laplace estimator, which is a widely used statistic used for calculating risk, in order to keep only the branches that are really safe. A handful of clean matches is not enough; about fifty are.

**3 Held-out check**
And a held-out check. The trimming used some samples to decide which tests to drop. Checking those same samples is not enough, because that is how the branch was shortened. So we hide a slice first, trim on the rest, and then look at the hidden slice: when this branch matches, the full integer network must still agree. One disagreement there, and the branch does not enter the pool.

This is measured agreement, not a proof for every input. That residual is the tenth of a point you will see later.

*click*

---

## 15. Remaining-work score · 0:40

The third step is the remaining-work score. After trimming and vetting we still have too many branches, so we need a way to pick a short list to embed.

NN2Logic ranks a route by the multiplications an exit leaves undone. On a CNN that is the wrong question. Skipping the head looks cheap, even though the convolution has already run, and that is almost all of the work.

This thesis ranks the other way: how much work is still left if we take this exit.

**A match**
If a sample hits a route, we only count the convolution windows that route has to read before it can decide. That is the receptive field. The rest of the map is not computed. If several routes would hit, we count the cheapest one.

**A miss**
If no route hits, we count the full network, plus a little extra because we already tried the routes and failed.

The score is this average, in MACs, over the samples. It is not a Spike measurement. We pick the short list that makes that average smallest, and then we generate the C from that fixed list.

A published certificate that still names a head test scores almost a full pass. That is why it does not move the mean.

*click*

---

## 16. Hybrid code generation · 0:45

So once we have selected the routes that are going to be checked before inferring the whole network, we emit the hybrid C code with this structure.

**HOT**
As we can see in the figure, we first execute the hot block, that is basically the calculation of the activations needed for the routes. In order to do so, a sparse loop is implemented that is better explained in the next slide.

**MATCH**
Then we compare the sign tests with the signs of the preselected routes, and if it matches, the program returns the class, saving the rest of the computation.

**MISS**
If it does not match any route, the cold block is executed. That is not the full network from scratch: it is a loop that finishes the convolution without calculating again the activations already calculated in the hot block, and then the remaining layers run, with the same weights.

*click*

---

## 17. Hot-loop emission · 0:50

The default writing stores the kernel once and applies it at the hot positions. Sharing buys bytes. It also delays the earliest return, because every hot neuron is computed before the first test.

**Figure**
Blue cells are hot: the routes need them. Pale cells wait for a miss.

**Right panel**
The kernel is stored once and applied at the hot positions. Border positions stay unrolled, because one shared body cannot shorten the window. The unrolled writing of the same routes is measured beside this. That trade is the emission table in the evaluation.

*click*

---

## 18. Evaluation · 0:05

Evaluation.

*click*

---

## 19. Workloads and reference · 0:50

Four networks, one matched looped reference. Fashion and Pneumonia share the stack, so a cycle difference between them is a difference in the images. Every convolution has one input channel. The arrows are the linear layers. Hot is the number of first-stage neurons the routes actually read. Ref. acc. is the looped integer network on the test set.

**Fashion-MNIST**
14×14, one channel, one 3×3 filter, average pool. Linear 49 to 4 to 2. Eleven routes, twenty hot neurons. Reference accuracy 97.65%.

**PneumoniaMNIST**
The same stack, normal versus pneumonia. Twelve routes, twenty-five hot. Reference accuracy 84.74%.

**MNIST even/odd**
28×28, one channel, one 5×5 filter, stride 2, average pool. Linear 49 to 4 to 2. Six routes, fifteen hot. Reference accuracy 91.88%.

**HAR, 3-class**
128×9, one channel, one 5×3 filter, average pool. Linear 12 to 8 to 3. One route, two hot. Reference accuracy 83.20%.

**PLATFORM**
The evaluation runs on Spike, the RISC-V simulator. The ISA is RV32IM, the integer set with multiply that Ibex implements. Architectural mcycle. No cache. No Flash wait states. A cycle count compares the two programs; it does not predict one chip.

**REFERENCE = HYBRID FALLBACK**
Same weights, same loop, same 32-bit accumulators. The measured difference is the routes.

*click*

---

## 20. Cycle results · 1:30

The hybrid is faster on all four networks. Cycles are counted on Spike, under the RV32IM ISA that Ibex implements. The average includes the misses.

**Fashion**
22.3% mean cycle reduction. Exit mass 36.7%. Fashion moves the mean because more than a third of the test set matches.

**Pneumonia**
8.4%. Exit 17.0%. Same graph, smaller exit mass.

**MNIST even/odd**
6.0%. Exit 12.2%. Exits less often under a heavier front end.

**HAR**
7.9%. Exit 7.7%. One short route. The mean still falls because that route skips a front end that holds most of the MACs, and because partial convolution shortens the path every sample starts.

**Table**
A match returns in 82–94% fewer cycles — Hyb. min against Ref. mean. The mean lies below that because it includes the misses. Accuracy: −0.05, −0.12, −0.08 pp on the image models, and 0.00 on HAR.

*click*

---

## 21. Cycle distributions · 1:00

The mean is a mixture of two modes. A cycle win is not, by itself, evidence that a route fired.

**Figure · Fashion-MNIST**
The low-cycle mode is a match. The mass near the reference is a miss that has already paid the dispatcher. Mean 22.3%. Exit 36.7%.

**Right panel**
The same shape on all four: Fashion 22.3% at 36.7% exit, Pneumonia 8.4% at 17.0%, MNIST 6.0% at 12.2%, HAR 7.9% at 7.7%. On HAR every test input can be faster than the reference while only 7.7% take the tree route: partial convolution shortens the hot path that every sample starts.

*click*

---

## 22. Code size · 1:00

Object code is text plus read-only data. Sharing the hot path is what keeps the image models near twice the looped reference.

**Left table**
2.16×, 2.15×, 2.11×, and 1.85×.

**Same routes, three writings · MNIST**
Unrolled: 7.4% at 12238 B. Shared loop, two routes in front: 6.9%. Shared loop, cut at zero — the reported writing: 6.0% at 6414 B.

**Figure**
Across route subsets of one tree, object code is linear in the number of hot first-stage MACs: about 15 bytes per MAC.

SKIP if late:
On Fashion the same sharing is 12% smaller and 354 cycles more — about 12× costlier per byte, because more than a third of the set exits.

*click*

---

## 23. Certificate and original selector · 1:00

The published machinery does not produce that early exit. Two different things sit on this slide: the origin of the tests, and the original ranking.

**Left · IIS**
The irreducible certificate is the smallest proving subset. On these CNNs it still names a head neuron, so the convolution runs in full. Reduction is 0.5% or less. Accuracy does not move.

**HAR row**
n/a, because the original selector is binary and HAR is three-class. One sentence if they notice the gap; it is not in the thesis.

**Bottom**
Accuracy stays exact. Neither IIS nor the original selector has an early-exit mode. The remaining-work score is the column that moves the mean, at about twice the code.

*click*

---

## 24. Findings · 0:05

Findings.

*click*

---

## 25. Conclusions · 1:00

Four results.

**The hybrid is faster**
Mean Ibex cycles fall by 22.3%, 8.4%, 6.0%, and 7.9% against a matched looped reference.

**The saving is an early exit**
The receptive field of a few first-stage neurons, not a shortened run of the whole network. The mean tracks the exit mass.

**The byte side is about 2×**
Sharing the hot path keeps the image models near twice the looped object code.

**Accuracy stays within a tenth of a point**
The encoding is exact. The gap is the price of trimming tests into the first stage.

*click*

---

## 26. Limitations · 0:45

Four cards.

**Unchanged certifier**
Constant leaves come from the published pass. Several networks never yielded a first-stage route.

**No IIS on the routes**
The embedded paths are not an IIS certificate. They can lose a fraction of a point of accuracy against the reference.

**One convolutional channel**
Geometry recovery and the loop are written for a single first-stage channel.

**What to extend**
A multi-channel front end under the same paired comparison. Vetting is not a proof. Cycles are architectural.

*click*

---

## 27. Thank you · stop

Thank you. I am happy to take questions.

---

## If you run long

Cut in this order:

1. Fashion footnote on slide 22
2. The “two different things” opening on slide 23
3. The first limitations card, if they already have the existence point
4. The host-checker sentence on slide 16

## If they ask in the first minute

No cycle figures before slide 20, unless a question forces them.

Published NN2Logic, if they insist (not on the slides): occupancy up to 14.9%, no accuracy loss; fully connected even/odd MNIST 0.7%.

## Backup

**Accuracy**
The gap is placing the tests in the first stage, not unfolding. Looped and unrolled references compute the same function. QAT error is outside this comparison.

**MAC breakdown**
Non-zero products after folding. Fashion and Pneumonia: 76.9% in the first stage. MNIST: 91.5%. HAR: 93.9%.

**All cycle distributions**
Same two-mode shape on every network.

**Full cycle table**
Ref min equals Ref mean: the looped reference is essentially constant. Hyb max can exceed the reference because a miss has already paid the dispatcher.

**MNIST emission swap**
Same routes, same 12.2% exit, same 91.80%. The loop does not reduce arithmetic; it removes copies. Reference: 15274 cycles, 3042 B.

**Leaf survey**
Fashion-10: 0%. Occupancy CNN: 33.6%, still no first-stage route.

**Limits**
One input channel. HAR shapes supplied. Border stays unrolled. Vetting is not a proof. Single first-stage channel. Architectural cycles. Certifier unchanged.

**Contributions**
Pipeline around an unchanged tree builder. Remaining-work score and first-stage return. Looped emission reported next to cycles.

**Geometry recovery**
Enumerate kernel, stride, padding. Accept only after the whole matrix is rebuilt and every row matches. A layer that no geometry reproduces is recorded as dense.
