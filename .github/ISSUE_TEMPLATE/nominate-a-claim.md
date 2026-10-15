---
name: Nominate a claim
about: Suggest a published trading-AI result for the lab to re-test
title: "Nominate: <paper, short claim>"
labels: nomination
assignees: ''
---

<!--
The lab re-tests claims at three levels: L1 the authors' own code and setup,
L2 this repository's engine (signal at the close, fill at the next open, fees,
at least 5 seeds), L3 a later or different window. Nominations are scored with
the rubric below and the top one takes the next free monthly slot. Fill in what
you can; "unknown" is a fine answer. Please do not paste vendor price data.
-->

## Paper

<!-- Title, authors, venue and year, and a stable link (arXiv, DOI, SSRN, publisher). -->

- Title and authors:
- Link:

## The claim, verbatim

<!--
Copy the sentence or the table row exactly as published, with the table, figure
or page number and the version you read (for example "v3", "camera-ready").
Include the number that would be tested (return, Sharpe ratio, drawdown, IC,
hit rate) and what it is compared with.
-->

> 

- Where: <!-- Table 2, p. 7, arXiv v1 -->
- Metric, window and baseline stated by the authors:

## Code and data

<!-- Link to the authors' repository and a commit or release if you know one. -->

- Code:
- Data used by the authors (source, licence, whether it is still obtainable):
- Does the code run today? <!-- yes / no / not tried -->

## Why it matters

<!--
Who relies on this result: citations, GitHub stars, products or courses built on
it, copycat papers. One short paragraph.
-->

## Scoring hints (optional)

<!--
Score each 1 to 5 if you want to; maintainers score every nomination themselves.

| Criterion | Weight | 5 means |
| --- | --- | --- |
| Claim quality | 30% | Peer-reviewed, concrete numbers, stated window and baseline, code available |
| Protocol fit | 25% | Our instruments can test it fairly with licensable data and affordable compute |
| Reach | 20% | Large existing audience (stars, citations) |
| Novelty | 15% | Nobody has published a multi-seed or out-of-sample re-test |
| Cost | 10% | Cheap in compute, API spend and maintainer time |
-->

- Prior re-tests you know of (links):
- Anything that would make a fair test hard (private data, closed models, live-only results):

## Your relation to the work

<!--
Please say which applies; it does not disqualify a nomination.
-->

- [ ] I am an author or co-author of the paper
- [ ] I maintain or contributed to the code
- [ ] I am affiliated with the authors' group or company
- [ ] I have no connection to the authors
- [ ] Other:

<!-- If you are an author you will get the same draft-and-reply notice as any author. -->
