# Quick Guide: Using the RAG Chatbot for Operators

This document explains how operators and engineering personnel can use  
the RAG chatbot to search documentation.

## 1. Purpose of the System

The RAG chatbot is intended for:

- quick search of instructions and regulations;
- clarifying maintenance parameters;
- finding report templates and formatting standards.

The system does **not** replace official documentation and orders.  
It is a convenient search assistant.

## 2. Typical Queries

It is recommended to formulate queries as if you are asking a colleague:

- “What is the maintenance regulation for pump unit N-250?”
- “What should be checked before starting the unit after repair?”
- “What are the safety requirements for working in the pump hall?”
- “Give me the structure of a technical incident report.”

It is useful to specify:

- equipment type (pump, compressor, tank);
- unit designation (N-250, K-10, etc.);
- type of operation (start-up, shutdown, maintenance, testing).

## 3. System Limitations

1. The chatbot is trained only on internal documentation loaded into the index.
2. If a document has not been added to `data/raw` and indexed, the chatbot “does not know” about it.
3. The chatbot’s answer must be verified against current official documents when:
   - regulations are updated;
   - new orders/instructions are issued;
   - work is performed under non-standard conditions.

## 4. Dialogue Examples

**Example 1**

> Query:  
> “How often should PM-2 be performed for pump unit N-250?”

Expected answer (shortened):

> “According to the maintenance regulation for pump unit N-250,  
> extended PM-2 must be performed at least once per quarter…”

**Example 2**

> Query:  
> “Provide a template for a technical incident report.”

Expected answer:

> A list of main sections: general information, brief description,  
> parameters, personnel actions, cause analysis, measures, and attachments.

## 5. Feedback and Quality Improvement

If the chatbot:

- provides conflicting answers;
- fails to find information that is definitely present in documentation;
- refers to outdated versions of regulations,

you should:

1. Capture the query/response example;
2. Send it to the person responsible for maintaining the RAG system;
3. If needed, update the documents in `data/raw` and rebuild the index.

## 6. Responsibilities

- Final decisions on technological operations are made based on:
  - approved regulations;
  - instructions from responsible engineers and managers.
- The RAG chatbot is a supporting tool and must not be treated as the only source of decisions.
