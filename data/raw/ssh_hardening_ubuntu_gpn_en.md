# Guideline: Basic SSH Hardening on Ubuntu Servers

Draft document for internal IT / OT perimeter.  
Purpose: define a minimum security baseline for SSH access to production servers.

## 1. Scope

This document applies to:

- bastion hosts;
- application servers;
- auxiliary services (logging, monitoring),

running Ubuntu Server 20.04 or higher.

## 2. General Principles

1. SSH access is allowed **only** with keys.
2. Password authentication is disabled.
3. SSH access is restricted by IP (VPN / jump-host).
4. All actions are logged locally and on a central log server.

## 3. Basic SSH Configuration Requirements

Configuration file: `/etc/ssh/sshd_config`.

Minimum set of parameters:

- Disable direct root login:

```text
PermitRootLogin no

```

- Disable password-based authentication:

```text
PasswordAuthentication no
ChallengeResponseAuthentication no

```

- Restrict protocol:

```text
Protocol 2

```

- Enable detailed logging:

```text
LogLevel VERBOSE

```

- Restrict user list (example):

```text
AllowUsers admin_ops deploy_ci

```

After changing the configuration, restart the SSH service using standard OS tools (systemd).


## 4. SSH Keys Requirements

1. Keys must be generated on the user’s workstation.

2. Private keys must remain on the user side only; sending them via open channels is prohibited.

3. Recommended key types: ed25519 or rsa with length at least 4096 bits.

4. The private key must be protected with a passphrase.


## 5. Network Access Restrictions

Network access to the SSH port (typically 22/tcp) must:

- be closed from external networks;

- be opened only for:

- VPN gateways;

- jump-host servers;

- trusted administrative subnets.

Firewall and access control rules are described in a separate network security policy.


## 6. Monitoring and Audit

1. SSH logs must be forwarded to a central log server.

2. Minimum set of monitored events:

- successful and failed login attempts;

- sshd configuration changes;

- changes in user lists.

3. A periodic review is required (at least once per quarter) for:

- active keys;

- accounts with SSH access.


## 7. Responsibilities

- The server administrator is responsible for correct sshd configuration and security updates.

- The department manager is responsible for maintaining the list of users who require SSH access for business purposes.

