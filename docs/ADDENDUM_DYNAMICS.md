# Addendum, Does the bistable attractor exist? A dynamical-systems audit of RAC

**Research Use Only.** This addendum interrogates the load-bearing premise of the Regulatory
Attractor Control (RAC) method: that the fitted system is a *bistable dynamical system* whose
essentiality score measures *collapse to a dead attractor*. It concludes, from four independent
model-free tests plus a cross-checked eigenvalue analysis, that **the premise does not hold at the
operating point**, and it explains, mechanistically, the phenomenological retraction already
recorded in [REVIEW_RESPONSE.md](../REVIEW_RESPONSE.md) §1/§15/§15b.

Scripts: `scripts/dynamics_characterization.py`, `scripts/verify_dynamics_instability.py`.
Results: `results/dynamics_characterization.json`, `results/verify_dynamics_instability.json`.

---

## 1. Why this premise is load-bearing

The method's own documentation states the system is

```
x_{t+1} = sigmoid(gain * (W x_t + b)),  gain = 4.0
```

with a **viable high-activation attractor** (fit to the DepMap PDAC cell-state panel) and a stable
**dead low-activation attractor**, so that "node essentiality = the network-wide collapse induced
by clamping that node down" is a statement about *basins of attraction*. The project summary is
explicit that bistability is required: *"contractive systems have a unique fixed point → cannot
express collapse."*

Everything downstream inherits this premise:

- the **collapse essentiality score** (`collapse_scores`) is defined as travel toward the dead
  attractor under node clamping;
- the **control design** (`control_design`) selects targets that "move the attractor along the
  healthy direction";
- the surviving methodological defence, after the essentiality claim was retracted, was
  *"RAC provides intervention semantics over a bistable system."*

If the fitted system is not actually bistable, if its equilibria are not stable attractors, then
each of these needs to be re-described in terms of what the dynamics actually do.

A crucial observation motivates the audit: **the dead attractor is trained in, not observed.** The
fit adds a penalty `dead_weight * relu(sigmoid(gain*(dead0 @ W.t() + b)) - 0.25)` that keeps the
0.05-state from activating during fitting. A penalty that holds one point low while the weights are
learned is *not* the same as the fitted map possessing a stable low fixed point. That is an
empirical question, and it had never been asked.

---

## 2. Method, five probes, then triple verification

All probes use the primary configuration (400 nodes, τ = 0.4, seed 20260620), the same fit whose
collapse scores were analysed in §15, with the device forced to CPU for determinism. The five
characterization probes: (1) a fixed-point inventory over a battery of initial conditions; (2) the
destination of the explicit low ("dead") initial states; (3) the spectral radius ρ(J) at each
equilibrium; (4) a gain sweep for a bifurcation/hysteresis signature; (5) whether clamping the
highest-collapse nodes reaches a dead basin.

Because a shocking result is exactly where a metric bug hides (see §16 of the review response,
where a mis-specified direction metric inverted the canonical tumour suppressors), the instability
was then **verified three ways that use no eigenvalue at all**, plus a Jacobian cross-check:

- **A. Convergence fraction**, iterate the map to a long horizon from every real cell state and
  from random inits; count how many reach a true fixed point (max step < 1e-6).
- **B. Direct perturbation growth**, settle, perturb by ε = 1e-3, iterate; does the deviation grow
  (unstable) or shrink (stable)? No Jacobian involved.
- **C. Spectral radius, analytic vs finite-difference**, compare the closed-form Jacobian to a
  perturbation-measured one; agreement rules out a derivative bug.
- **D. Fixed-point quality**, the one-step residual `‖σ(gain·(Ws+b)) − s‖` of each cell state.

---

## 3. Result. The equilibria are not attractors, and there is no bistability at gain 4.0

| test | result | interpretation |
|---|---|---|
| **D** fixed-point quality | one-step residual **median 0.729, max 0.980** | the cell states move ~0.73 per node in a *single* step, they are nowhere near fixed points |
| **A** convergence | **0 / 54** cell-state inits and **0 / 30** random inits converge (2000 iters, tol 1e-6) | the map has no reachable stable attractor |
| **B** perturbation growth | **median 3.41×**, max 71× over 20 steps | perturbations grow, the dynamics are expansive/unstable |
| **C** spectral radius | **ρ = 1.02** at the mean-state iterate (analytic 1.019, finite-diff 1.023, agree to 0.02); **viable ρ = 1.13, dead ρ = 1.05** at the clustered equilibria | every equilibrium tested is linearly unstable (ρ > 1), cross-checked |
| **4** bifurcation | hysteresis (gap > 0.05) only from **gain ≈ 5.9**; operating gain 4.0 is **below** it | at the operating gain there is no two-basin structure to begin with |
| **5** collapse mechanism | clamping reaches a dead basin for **0 / 10** probed nodes | "collapse to the dead attractor" reaches no dead basin, because none is stable here |

The evidence is convergent and, for A/B/D, entirely model-free, no eigenvalue is needed to see that
a system which moves 0.73/node per step and never converges from 84 initial conditions is not
sitting at a stable attractor. **The fitted RAC system, at its operating gain, is a non-converging,
linearly unstable map whose "viable" and "dead" attractors are not stable basins the dynamics
reach.**

### A metric bug, caught by the cross-check (and disclosed)

The first characterization run reported ρ ≈ 2.3–2.6. That was wrong: the analytic Jacobian
evaluated the sigmoid derivative at `z = Wx+b` instead of at the map's actual argument `gain·z`. The
finite-difference cross-check (probe C) disagreed by 1.9 elementwise, exposing the error; corrected,
analytic (1.019) and finite-difference (1.023) agree, and ρ ≈ 1.02. The conclusion is unchanged
because it never rested on the eigenvalue, A, B and D are model-free, but the *number* was wrong
and is corrected here instead of quietly. This is the same failure mode the review response
catalogues throughout, caught this time by a guard placed specifically to catch it.

---

## 4. What `collapse_scores` actually measures

`collapse_scores` calls `_settle` for 250 iterations, then compares the result with and without a
node clamped. On a map that never converges, `_settle` returns a **250-step transient snapshot**,
not a fixed point. The collapse score is therefore a difference of two transients:

> how much does clamping node *i* change where the 250-step trajectory ends up, summed over the
> other nodes.

This is a deterministic, reproducible function of the graph and weights, but it is a **graph-influence
propagation** quantity, not a basin-transition quantity. And a propagation score over a
co-expression graph is expected to track node degree, because high-degree nodes influence more of
the transient.

**This is the mechanistic root of the §15 retraction.** §15 showed empirically that collapse carries
no essentiality signal beyond degree (partial ρ = 0.028, degree-matched AUC ≤ 0.49) and §15b showed
the apparent PDAC-selective signal was an artifact (collapse ranks KRAS, the strongest selective
dependency, at the 8th percentile). The dynamical audit explains *why*: the mechanism the score was
meant to exploit. Travel across a separatrix into a dead attractor, does not exist at the operating
point, so the score reduces to nonlinear graph propagation, which is degree-like. The phenomenology
(§15) and the mechanism (this addendum) are one story.

---

## 5. Scope of the correction, what changes and what does not

**Must be retired:**
- the description of RAC as a *bistable attractor-control* system;
- "essentiality = collapse to the dead attractor", the essentiality claim was already retracted
  (§15); its stated *mechanism* is now also withdrawn;
- "intervention semantics over a bistable system" as the surviving methodological defence, the
  semantics are over an unstable, non-converging map;
- `control_design`'s "move the attractor along the healthy direction", it moves a 250-step
  transient, not an attractor.

**Honestly re-described:** RAC is a **nonlinear graph-influence propagation score** over a fitted,
sign-anchored, co-expression-masked weight matrix. That is a legitimate network construction; it is
simply not what it was called, and (per §15) it does not outperform network degree.

**Untouched:** the signed **intervention gate** (§16) is independent of the dynamics, it is a
role × direction gate calibrated against TCGA copy number (11/12 roles corroborated), and stands.
The multi-omic data layers (CNA, methylation, CPTAC, Hi-C, ENCODE chromatin) and the off-target
repair (§13/§14) are likewise independent.

---

## 6. What would make the premise true (constructive), and a tested caveat

The audit suggests a *route* to a genuinely bistable system, should the method be rebuilt to match
its own description. **§8 tests the first item directly and shows it is necessary but not
sufficient. Raising the gain alone does not work, so the list is ordered by what §8 established.**

1. **Enforce fixed-point quality.** The cell states have RMS one-step residual ≈ 0.20 and max 0.73;
   they must be constrained to be *actual* fixed points (an enforced residual penalty, not an
   averaged one) before they can be called attractors. §8 shows this is the binding constraint: at
   no gain in [4, 8] does the current fit converge.
2. **Enforce stability, not just low activation.** The dead state must be trained to be a *stable*
   fixed point (ρ < 1 in its neighbourhood), e.g. by penalising the local spectral radius, not
   merely its activation level.
3. **Operate above the bifurcation**, *necessary but not sufficient*. Hysteresis appears only from
   gain ≈ 5.9, above the operating 4.0; but §8 shows that simply refitting at higher gains (5–8)
   still yields **0/54 convergence at every gain**, because the fit does not enforce (1) or (2).
   Gain must exceed the bifurcation *and* the fit must produce true stable fixed points.
4. **Verify, don't assume.** Ship the convergence / perturbation-growth / spectral-radius checks in
   the test suite so "bistable" is a property measured on every fit, not asserted once.

Until those hold, the honest name for the method is a graph-influence score, and, as §15 and §7
both establish, it does not beat degree.

---

## 7. Can raising the gain rescue it? No, the failure is structural on both axes

§6 floated running above the bifurcation. `scripts/gain_sweep_rescue.py` puts that to the test. RAC is
refit at gains 4 through 8, each fit tuning W and b for its own gain, and every fit checked for
convergence, spectral radius, and whether collapse beats the fixed degree reference of 0.629. What
counts as a rescue was fixed up front. A settled, stable regime, and collapse above degree. Both, or
it fails.

| gain | converged fraction | ρ | collapse AUC | degree AUC | ΔAUC | beats degree? |
|---|---|---|---|---|---|---|
| 4.0 | 0.00 | 1.02 | 0.606 | 0.629 | −0.023 | no |
| 5.0 | 0.00 | 1.26 | 0.572 | 0.629 | −0.057 | no |
| 6.0 | 0.00 | 1.34 | 0.586 | 0.629 | −0.043 | no |
| 7.0 | 0.00 | 0.96 | 0.674 | 0.629 | **+0.045** | (isolated) |
| 8.0 | 0.00 | 0.92 | 0.615 | 0.629 | −0.014 | no |

**Neither axis is rescued.**

*Bistability axis.* Convergence is **0.00 at every gain** and not one refit produces a map that
settles. Even where ρ < 1, at gains 7 and 8, the map still does not converge, and with nothing
converging ρ is measured at a transient snapshot, not at a fixed point, so it certifies
nothing. The characterization's bifurcation at 5.9 is hysteresis between two non-converged transient
means, not two stable basins. Raising the gain alone does not yield a bistable system with
this fit. Enforcing true stable fixed points is the binding requirement.

*Prediction axis.* Collapse never robustly beats degree. Four of five gains are negative, and the
lone positive at gain 7 is isolated with both neighbours negative, sits in a non-convergent regime,
and is well inside §15's noise band of ±0.11. It is reported, not claimed, because treating it
as a rescue would mean selecting one gain out of five on the outcome, the exact error §15 was built to
avoid.

**Conclusion. The §15 retraction is structural, not a tuning artifact.** Dynamics can only propagate
what the graph encodes. Essentiality is not in the co-expression graph beyond degree, as §15 and §15b
showed with a partial ρ of 0.028 and KRAS at the 8th percentile, so no reparameterisation of the
dynamics over that graph can manufacture it. The failure sits in the substrate, not the operating
point. RAC as posed is not one hyperparameter away from working, and a genuine rebuild would have to
change what the graph encodes, through directed or causal edges and perturbation data, and not how
the dynamics run over it.

---

## 8. One-line summary

The RAC system is not bistable at its operating point *or at any gain in [4, 8]*: its equilibria are
unstable and non-converging (0/84 inits converge at gain 4; 0/54 at every swept gain; perturbations
grow 3.4×; ρ ≈ 1.02; one-step residual 0.73). The "attractor-collapse" essentiality score is a
nonlinear graph-influence propagation quantity, which is the mechanistic reason it does not
outperform network degree. And raising the gain does not change that, so the §15 retraction is
structural (in the graph), not a tuning artifact.
