# SSH Access Approval Policy

Draft document.  
Purpose: define a unified process for granting and revoking SSH access to production servers.

## 1. General Provisions

1. SSH access is granted only for business-related tasks.
2. All access requests must follow a formal approval workflow.
3. Any SSH access must be **personal** and **named**:
   - the use of shared technical accounts without a responsible owner is prohibited.

## 2. Roles and Responsibilities

- **Access requester** — employee who needs access to perform tasks.
- **Line manager** — confirms the business need for access.
- **IT/IS service** — evaluates justification, risks, and compliance with security policies.
- **System administrator** — technically creates/updates accounts and access keys.

## 3. Access Request Procedure

1. The requester creates a ticket in the Service Desk (or similar system), including:
   - purpose of access (what tasks will be performed);
   - list of target servers or groups;
   - type of access (log reading, deployment, administration);
   - requested duration of access.
2. The line manager approves or rejects the request.
3. IT/IS service:
   - checks whether the requested access follows the “least privilege” principle;
   - suggests narrowing the scope or duration if needed.
4. After approval, the administrator:
   - creates/updates the user account;
   - registers the user’s public key;
   - assigns the user to the required groups or roles.

## 4. Access Duration and Review

1. SSH access must be **time-bound**:
   - “until revoked” is allowed only for permanent administration roles.
2. Recommended durations:
   - project / temporary activities — up to 3 months;
   - regular admins — up to 12 months, with mandatory yearly review.
3. Access review must be performed at least quarterly:
   - identify unused accounts;
   - remove access for employees who left or changed position;
   - update the list of servers where access is needed.

## 5. Prohibited Practices

The following is prohibited:

1. Sharing SSH keys with third parties (including colleagues).
2. Using shared keys for multiple employees.
3. Using SSH access for non-business purposes.
4. Changing system settings without a ticket / change record.

## 6. Access Revocation

1. Access must be revoked immediately in case of:
   - employee termination or role change;
   - violation of information security requirements;
   - suspected or confirmed key compromise.
2. IT/IS service must:
   - document the reason for revocation;
   - initiate an incident investigation if necessary.

## 7. Logging and Control

1. All SSH sessions should be logged (command audit when feasible).
2. Log data is stored according to the corporate log retention policy.
3. SSH logs are a key information source during incident investigations.
