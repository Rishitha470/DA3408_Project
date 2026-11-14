# AI Use Disclosure


I used AI assistants while building this project. This file states which tools I used, what they produced,
and what I did myself, so that the work can be assessed fairly.

## Tools

| Tool | Used for |
|---|---|
| Claude  | generating the first version of the code base and documents |
| Claude Code  | diagnosing problems, code fixes and documentation updates |

## What the AI tools produced

* **First version of the code base (Claude):** I used this to build the first version of the project so that I can get an idea of what to do.
* **Diagrams:** the architecture diagram (`docs/architecture.md`) and the high-level design diagram
  (`docs/HLD.md`) were generated with Claude as Mermaid code, which GitHub draws as pictures. I reviewed
  both against the running system to make sure every block and arrow matches how it actually works.
* **Fixes and changes during my runs:** when my runs showed problems, it diagnosed them and
  changed the code: the non-reproducible data split, the drift window size, robust model registration, the
  Grafana panel colours and the error-rate rule, the registry alias during rollback, the MLflow server workers,
  line-ending settings, and two new tests. These are listed in [REPORT.md](REPORT.md) section 12.
* **Documentation:** it helped me to analyse the results and in writing the report and in debugging and helped in documenation of the repo.


