# Incident Response Runbook

## Severity levels
- SEV1: customer-facing outage or data breach. Page the on-call lead immediately and open a P1 ticket.
- SEV2: major feature degraded for many users. Open a P2 ticket within 15 minutes.
- SEV3: minor issue with a workaround. Open a P3 ticket during working hours.

## First 15 minutes
1. Acknowledge the page and join the incident channel.
2. Assign an incident commander and a communications lead.
3. Check the Cloud Monitoring dashboard for error rate, p95 latency and saturation.
4. If a deploy happened in the last 2 hours, roll back first and investigate second.

## Rollback on Cloud Run
Shift 100% of traffic to the previous healthy revision with `gcloud run services update-traffic SERVICE --to-revisions=REVISION=100`. Rollbacks do not need change-board approval during a SEV1 or SEV2.

## Post-incident review
A blameless post-incident review is due within 5 working days for every SEV1 and SEV2, with action items tracked as tickets.
