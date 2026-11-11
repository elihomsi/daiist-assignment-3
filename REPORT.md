# Assignment 3 Report

- **Name**: Eli Homsi
- **Student ID**: 18938
- **Group**: BBADBA 5A

## Use case

**Who asks.** Financial analysts and investors who read companies' annual reports. The system answers only from the annual reports indexed in it: Apple, NVIDIA and Tesla to start with. An analyst can add any other company's annual report with the add-document tool, and a question about a company that is not indexed gets "I don't know".

**What they do with the answer.** They ask about a company's business, its financial performance and its risk disclosures, and use the answers in their notes and investment decisions.

**What a wrong answer costs compared with "I don't know".** An "I don't know" costs the reader about 10 minutes of looking the answer up in the filing, roughly $12 at an assumed $75 an hour. A wrong answer that is caught later costs about 2 hours of rework, roughly $150. A wrong answer that is not caught can lead to a poor investment decision. A guess is only worth giving if it is right more than about 92% of the time, so when the filings do not contain the answer, the assistant should say which information is missing instead of estimating or inventing a figure.

## Corpus

*Table: document, URL, date, licence, words.*

## Ingestion

*How you extract, clean and chunk; one extraction problem you fixed; chunk metadata; number of chunks.*

## Test set

*How you wrote it, counts by kind, two examples.*

## Retrieval

*Hit rate at 1/3/5 per design, words added to the prompt, the design you chose.*

## Tools and subagent

*Your tools' descriptions, one search result, the subagent's instructions, one transcript from your editor.*

## Adding a document

*What happens when a document is already indexed. The unanswerable question, the document you added, the answer before and after.*

## Harness and guardrails

*Your AGENTS.md rules, the approval step, and your extra guardrail with one case where it fired.*

## Evaluation

*Code checks, the judge's rubric, your agreement with the judge, and results: correct, unsupported, correct refusals, wrong refusals.*

## Failures

*Failed questions by cause, with counts, and what to fix first.*

## Decision

*Would you let users rely on it, for what, with which safeguard? Cost and latency per question.*

## AI use

*What you used and how, or "none".*
