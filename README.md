# DLDP-BF
DLDP-BF_2025

## File structure
  \item \textbf{Non\_Privacy}~\cite{bloom1970space}: The Non\_Privacy method serves as a baseline, where a Bloom Filter uses a fixed-length bit array and multiple independent hash functions to map elements to specific positions and set them to 1, without introducing any noise into the bit array. This method relies entirely on deterministic insertion and query rules to perform membership tests, but it provides no privacy protection.
    \item \textbf{RAPPOR}~\cite{erlingsson2014rappor}: A randomized response-based approach that introduces noise proportional to the number of correlated records. It enables client-side local differential privacy by perturbing each bit in a Bloom filter representation of the input.
    \item \textbf{DPBloomFilter}~\cite{ke2025dpbloomfilter}: A method that injects noise into the Bloom filter based on correlated sensitivity analysis, aiming to optimize the trade-off between utility and privacy under the differential privacy framework.
    \item \textbf{DLDP-BF (Proposed)}: We propose A Differentiated Local Differential Privacy Bloom Filter for Membership Queries (DLDP-BF). 
    It dynamically allocates hash functions and privacy budgets according to the importance of data, aiming to achieve high utility under rigorous local differential privacy guarantees.
