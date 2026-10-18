# Evaluation report

Mode: `offline` | Questions: 49 | Gated prompt: `v2_cited`

| prompt | retrieval_hit_rate | answer_accuracy | faithfulness | hallucination_rate | abstention_accuracy | false_abstention_rate | access_leakage |
|---|---|---|---|---|---|---|---|
| v1_baseline | 0.975 | 0.9 | 1.0 | 0.0 | 1.0 | 0.075 | 0 |
| v2_cited | 1.0 | 0.925 | 1.0 | 0.0 | 1.0 | 0.075 | 0 |
| v3_concise | 0.975 | 0.875 | 1.0 | 0.0 | 1.0 | 0.125 | 0 |

## Gate

PASS

## Failures: v1_baseline (4)

- **q09** (answerable, role=employee) Which days am I expected to work from the office?  
  answer: I don't have enough information in the knowledge base to answer that.  
  retrieved: ['remote_work_policy.md', 'employee_handbook_excerpt.md', 'leave_policy.md']
- **q13** (answerable, role=employee) What is the minimum password length?  
  answer: I don't have enough information in the knowledge base to answer that.  
  retrieved: ['it_security_policy.md', 'leave_policy.md', 'remote_work_policy.md']
- **q29** (answerable, role=employee) Who does an unacknowledged page escalate to?  
  answer: I don't have enough information in the knowledge base to answer that.  
  retrieved: ['engineering_oncall_handbook.md', 'engineering_oncall_handbook.md', 'remote_work_policy.md']
- **q31** (answerable, role=employee) Within how many days must I finish mandatory training?  
  answer: Expense claims must be submitted within 30 days of the expense date. [expense_policy.md]  
  retrieved: ['it_security_policy.md', 'engineering_oncall_handbook.md', 'expense_policy.md']

## Failures: v2_cited (3)

- **q09** (answerable, role=employee) Which days am I expected to work from the office?  
  answer: I don't have enough information in the knowledge base to answer that.  
  retrieved: ['remote_work_policy.md', 'employee_handbook_excerpt.md', 'leave_policy.md', 'remote_work_policy.md']
- **q13** (answerable, role=employee) What is the minimum password length?  
  answer: I don't have enough information in the knowledge base to answer that.  
  retrieved: ['it_security_policy.md', 'leave_policy.md', 'remote_work_policy.md', 'remote_work_policy.md']
- **q29** (answerable, role=employee) Who does an unacknowledged page escalate to?  
  answer: I don't have enough information in the knowledge base to answer that.  
  retrieved: ['engineering_oncall_handbook.md', 'engineering_oncall_handbook.md', 'remote_work_policy.md', 'employee_handbook_excerpt.md']

## Failures: v3_concise (5)

- **q09** (answerable, role=employee) Which days am I expected to work from the office?  
  answer: I don't have enough information in the knowledge base to answer that.  
  retrieved: ['remote_work_policy.md', 'employee_handbook_excerpt.md']
- **q13** (answerable, role=employee) What is the minimum password length?  
  answer: I don't have enough information in the knowledge base to answer that.  
  retrieved: ['it_security_policy.md', 'leave_policy.md']
- **q19** (answerable, role=employee) Above what amount do I need a receipt?  
  answer: I don't have enough information in the knowledge base to answer that.  
  retrieved: ['expense_policy.md', 'expense_policy.md']
- **q29** (answerable, role=employee) Who does an unacknowledged page escalate to?  
  answer: I don't have enough information in the knowledge base to answer that.  
  retrieved: ['engineering_oncall_handbook.md', 'engineering_oncall_handbook.md']
- **q31** (answerable, role=employee) Within how many days must I finish mandatory training?  
  answer: I don't have enough information in the knowledge base to answer that.  
  retrieved: ['it_security_policy.md', 'engineering_oncall_handbook.md']
