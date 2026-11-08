# E026 diagnostic: estimand and limits

For a fixed development task, let d index four aspects and i the 18 visible options for an aspect. Two provisional preference predicates produce T/F/U. Let z_di be their conjunction: 1 for T, 0 for F, unresolved in [0,1] for U. This target excludes route feasibility, price optimization, and unrepresented real-world properties.

The baseline executor uses p0_di = 1/18. A fixed aspect question reveals the specified subset of that aspect's two predicate definitions. It removes only options with a revealed F, retains U, and uses p1_di uniformly on the remaining options. If no option remains, p1=p0 and the fallback is recorded. Other aspects retain p0.

The task score is the mean of four aspect scores. Conditional on this fixed task and information mask, the gain is

    Delta(q, mask) = (1/4) sum_i (p1_di - p0_di) z_di.

There is no expectation over user answers or hidden-goal priors here. The expectation is solely over the executor's randomized action. The scorer has privileged access to the task's complete preference predicates; question rankings are oracle diagnostics.

For c_i=(p1_di-p0_di)/4, coupled completion bounds are

    L = sum_{T} c_i + sum_{U} min(c_i, 0)
    U = sum_{T} c_i + sum_{U} max(c_i, 0).

The same z_di is used before and after clarification. Subtracting independent score endpoints can produce unnecessarily loose or misleading gain bounds. Independent completions may overapproximate feasible shared semantic interpretations, so these bounds can be conservative. They are not sampling confidence intervals.

The independent verifier uses a different counting derivation. Filtering cannot remove any T/U conjunction: anything removed has at least one F and hence an F conjunction. If t is the number of T options, u the number of U options, k the retained count and n=18, every completion with j successful unknown options has

    Delta_j = ((t+j)/k - (t+j)/n)/4, for j=0,...,u.

Enumerating j covers every possible value among all binary completions because every unknown surviving option has the same coefficient. The extrema must match the saved rational bounds. Separate unit tests check the generic coefficient formula against all binary completions for all 81 possible four-element T/F/U assignments, including negative coefficients on U.

A robust winner has L(q)>U(r) for every r!=q. A ranking reversal requires unique robust winners under both full and the specified partial mask, with different IDs. Ties and overlapping intervals have no robust winner. Both partial masks are reported separately. No inference about population prevalence or statistical significance is made.

The mask reveals both source preference positions, only position 0, only position 1, or none. These are deterministic ablations of information access. They do not establish P(answer|question, context), realistic nonresponse behavior, natural-language extraction accuracy, or an actionable policy without access to the complete profile.

The four generic question texts are fixed independently of which preference categories are active. They are not passed through UserBench's original question router; acceptance of these broad questions by that simulator has not been tested. Source ordering is not behavioral evidence.

Cost of clarification is held equal across questions and is not estimated. Costs of travel options, hidden feasibility constraints and the original intermediate rewards are not part of this diagnostic. Original source-best/source-correct labels are retained for comparison, never used to fit the rules.

The report's example and figure were selected after observing the fixed sample's outcomes to explain the first robust reversal. The two reversal tasks share flight:2-26; they cannot be treated as independent replication. E026 does not establish publication-level novelty.
