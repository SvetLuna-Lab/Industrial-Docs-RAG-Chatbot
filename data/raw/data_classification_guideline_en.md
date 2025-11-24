# Data Classification Guideline

Draft document.  
Purpose: define basic data classification levels and handling rules  
for use in RAG systems and internal services.

## 1. Classification Levels

It is recommended to use at least four levels:

1. **Public data**  
   - information officially published outside the company;
   - examples: press releases, marketing materials.

2. **Internal data**  
   - information for internal use within the company;
   - examples: generic internal instructions, training materials.

3. **Confidential data**  
   - data that could harm the company if disclosed;
   - examples: commercial terms, parts of technical documentation, network diagrams.

4. **Highly confidential data**  
   - critical information with strict protection requirements;
   - examples: encryption keys, passwords, detailed IT/OT infrastructure diagrams.

## 2. General Data Handling Rules

1. A classification level must be assigned **when the document is created** or first registered.
2. When a document is copied or forwarded, its classification level must be preserved.
3. It is prohibited to downgrade the classification level without approval from the data owner.

## 3. Classification for RAG Systems

When preparing a corpus for RAG:

1. By default, the index may include:
   - public data;
   - internal data (assuming basic access control is in place).

2. For confidential and highly confidential data, the following is required:

   - a separate index and/or separate environment;
   - user authentication and authorization;
   - logging of all access to such data.

3. Categories that **must not** be added to the general RAG index:

   - authentication data (passwords, keys, tokens);
   - personal data without a clear legal basis for processing;
   - any materials explicitly restricted by internal or external regulations.

## 4. Marking Classification Level in Documents

It is recommended to indicate the classification level:

- at the beginning of the document (header);
- as a separate field in document management systems (e.g., `classification_level`).

Examples:

- `[INTERNAL] Maintenance Regulation for Pump Unit N-250`
- `[CONFIDENTIAL] Network Infrastructure Diagram for PS-1`

## 5. Responsibilities

- The data owner is responsible for correctly assigning the classification level.
- Users are responsible for following data handling rules:
  - no unauthorized copying;
  - no sending outside the company without proper approval.
