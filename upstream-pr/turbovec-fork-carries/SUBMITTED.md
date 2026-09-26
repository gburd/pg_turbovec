# turbovec upstreaming — SUBMITTED 2026-09-26

Filed as issues (PRs are invitation-only per turbovec CONTRIBUTING; the issues
request contributor access and link ready-to-pull branches):

| carry | issue | branch (off 1.0.0 ccab9f3) | risk |
|---|---|---|---|
| #3 parallel repack (byte-identical) | https://github.com/RyanCodrai/turbovec/issues/545 | gburd:pr-parallel-repack (stacked on pub) | low |
| #1 pub pack::repack | https://github.com/RyanCodrai/turbovec/issues/546 | gburd:pr-pub-repack | low-med |
| #2 IdMapIndex parts API | https://github.com/RyanCodrai/turbovec/issues/547 | gburd:pr-parts-api | med (widest surface) |

Compare links posted as issue comments. PR-create was blocked (gburd lacks
collaborator access — expected, invite-only). Branches are pushed to
github.com/gburd/turbovec and pullable directly; open the PRs once access is
granted. Each branch carries a CHANGELOG `## [Unreleased]` line (the gate) and
#545 carries the byte-identity test (the mutation gate).

Still TODO before/if a PR is opened: re-verify #545's 250k micro-bench on AVX2
(the end-to-end 1766→566ms cold-scan is already confirmed). #547's final shape
(struct vs positional ctor) is a design conversation to have on the issue first.
