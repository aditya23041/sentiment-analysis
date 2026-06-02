import json

with open('jobs.json', encoding='utf-8') as f:
    data = json.load(f)
failed_job = next(j for j in data['jobs'] if j['conclusion'] == 'failure')
failed_step = next(s for s in failed_job['steps'] if s['conclusion'] == 'failure')
print(f"Job: {failed_job['name']}")
print(f"Failed Step: {failed_step['name']}")
