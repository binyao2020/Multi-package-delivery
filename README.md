# Multi-package-delivery

We consider multi-package delivery problem (more details refer to ./doc/description.pdf). This project models the problem and provides a routing algorithm based on column generation. The subproblem is formulated as ESPPRC (Elementary Shortest Path Problem with Resource Constraints) and solved by dynamic programming. We tested the algorithm on 3 randomly generated instances for each scale and evaluated the results with Clarke-Wright saving heuristics. The results are provided in ./doc/report.pdf.
