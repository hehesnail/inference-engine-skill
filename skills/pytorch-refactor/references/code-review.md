# Reviewing LMDeploy PyTorch changes

## First pass

Pin the requested target. For a PR or branch, inspect the diff from the merge
base with its intended target to the reviewed head. For a commit or local
changes, use the comparison the user requested. Keep implementation changes
and publication separate from a review request.

Read the complete scoped diff, including tests. Establish the intended change
from the request and existing caller contracts. For a fix, trace the failing
input to the first wrong state. For a refactor, follow an existing caller across
the changed boundary. For an optimization, verify how the workload selects the
new path. Use the topic references to check the contracts you encounter, and
keep looking for ordinary logic errors outside those references. When a
finding depends on a repository rule, check the instructions that cover the
changed path and cite the specific requirement.

For a change that crosses owners, read it from both ends. Trace what the
producer now emits, then start at an actual consumer and work backwards to
what it expects. Pick the expensive failure the first pass could miss. For
example, a scheduler test may show that a block was freed while the worker
can still read its contents. For a local change with one caller, a focused
source check and the existing test can settle the question.

Inspect the author's evidence at the boundary that matters. Identify where a
fixture replaces a real component. A successful HTTP response can establish
that validation accepted a request. If the claim concerns a template option,
inspect the rendered prompt. If it concerns a GPU implementation, verify which
backend ran. Reuse a matching result when it already answers the question.

## Match evidence to the failure

Start with a supported input or configuration and follow it to the observable
effect. Check what could invalidate the suspicion, such as an earlier guard, a
fallback selected on that device, or a value whose lifetime ends before the
changed step. Compare with the base when attribution is unclear, including
cases where the change first makes an older defect reachable.

Choose a check that distinguishes the suspected failure from correct behavior.
These examples guide the choice according to the affected contract.

| Change | Useful evidence |
|---|---|
| Model or backend dispatch | Follow a real supported configuration through selection to the implementation and weight layout it consumes. Check fallback if the change affects its reachability. |
| Batch or cache state | Track distinguishable requests through the affected transition, such as removing one row or rolling back a tentative allocation. Check what the next consumer sees for the surviving request. A prefill-only field can finish its lifetime before decode. |
| Distributed control flow or asynchronous work | For an early return, compare participating ranks at their next collective. For a buffer reused across streams, identify the producer, the last reader and the ordering between them. Exercise the relevant topology or streams when runtime behavior is in question. |
| Performance claim | Confirm the runtime selected the changed implementation. Compare the agreed workload and timing boundary with matching warmup and cache state. Use unprofiled runs for throughput and traces to explain where time went. |

A source-level demonstration can establish a deterministic defect. When a
conclusion depends on execution, use the smallest relevant check that reaches
the path and identify the tested revision and configuration. Reuse matching
evidence. If verification is incomplete, say which observation is still needed.

## Verify candidate findings

Treat a first-pass suspicion as a question. Before grading it, revisit the
changed lines from the affected caller's perspective. Look for the strongest
plausible explanation under which the code works, then check the source,
state transition or paired run that distinguishes the two explanations.
Separate the observed result from the premise needed to call it a defect.

An independent read is most useful when a costly conclusion rests on an
assumption, for example which rank owns a buffer during rollback. When an
independent reviewer is available and authorized, give them the pinned target,
the PR intent and the disputed question. Ask them to trace the path themselves
and return evidence for either outcome. A reviewer working alone can do the
same check from the consumer or against the base. Resolve disagreements with
the first point where the predicted behaviors differ.

## Report and re-review

Lead with confirmed findings in impact order. Anchor each to the smallest
changed location that explains it. Give the triggering scenario, the changed
behavior and its consequence, and the source or run that supports it. Combine
verified instances that share a cause and remedy. Separate material coverage
gaps from defects and name the behavior a missing test would distinguish.
If the scoped review yields no actionable defects, say so directly.

If the same input already fails on the base, check whether the patch makes it
reachable for additional callers or worsens its effect. Mention a relevant
pre-existing limitation separately when it helps interpret the result. For
architecture or scope choices with multiple valid answers, explain the
tradeoff to the maintainer. Reserve defect priority for concrete failures.

For a follow-up, identify the last reviewed head and the current head. Recheck
the original trigger, then inspect the revision and affected callers for new
regressions. Carry forward unresolved findings once. Widen the comparison if a
rebase, shared interface change or redesign alters earlier reasoning. If there
is no trustworthy previous review, review the current scoped diff in full.
Finish when the revisions and their effects are covered and each earlier
finding has a clear disposition.

If a revision fixes one instance of a shared defect, inspect sibling uses of
the same contract before asking for another round. If successive rounds dispute
the same expected behavior, summarize the evidence and identify the decision
the owner needs to make. Close a disproved finding with the reason once.

## Maintaining these instructions

When adding a review rule, name the contract, where it applies and a valid
exception. Try a violating diff, a similar safe change and an unrelated
change. Check that ordinary bugs outside the rule still surface. Put
deterministic checks in the relevant tests or lint configuration when working
on that code. Keep module-specific guidance in its existing topic reference.

Use confirmed escapes and consequential false positives to evaluate the
reviewer. Keep the original diff and a known-good comparison outside the
instructions, with the expected outcomes hidden during replay. Check whether
the reviewer catches the defect, leaves the clean case alone, or recommends
a harmful change. Use the outcome to choose between a regression test, a
narrower review rule, or a correction to that individual review.

For the original discussion of independent verification, repeat reviews and
incident replay, see [the Claude Code team's engineering account](https://claude.dev/blog/solving-code-review-at-anthropic/)
and [Anthropic's public review command](https://github.com/anthropics/claude-plugins-official/blob/main/plugins/code-review/commands/code-review.md).
