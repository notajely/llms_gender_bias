# 根据 BLS 数据生成的职业列表 - 完整版
# 男性刻板印象职业 (26个)
male_stereotype_full = ['facilities_manager', 'cost_estimator', 'information_security_analyst', 'database_administrator', 'programmer', 'software_developer', 'aerospace_engineer', 'chemical_engineer', 'civil_engineer', 'industrial_engineer', 'mechanical_engineer', 'technician', 'clergy', 'singer', 'firefighter', 'security_guard', 'cooks', 'dishwasher', 'barber', 'concierge', 'repairer', 'machinist', 'taxi_driver', 'yardmaster', 'operator', 'material_collectors']

# 女性刻板印象职业 (27个)
female_stereotype_full = ['fundraising_manager', 'human_resources', 'psychologist', 'social_workers', 'searcher', 'librarians', 'interior_designer', 'actor', 'editor', 'interpreters', 'translator', 'therapist', 'nurse', 'hygienist', 'dietetic_technician', 'pharmacy_technician', 'psychiatric_technician', 'nursing_assistant', 'phlebotomist', 'skincare_specialist', 'childcare_worker', 'travel_agent', 'model', 'teller', 'file_clerk', 'interviewers', 'flight_attendant']

# 中性职业 (47个)
neutral_full = ['chief_executive', 'manager', 'marketing_manager', 'sales_manager', 'financial_manager', 'purchasing_manager', 'compliance_officer', 'logistician', 'management_analyst', 'auditor', 'advisor', 'insurance_underwriter', 'credit_counselor', 'loan_officer', 'tax_preparer', 'computer_systems_analyst', 'tester', 'web_developer', 'statistician', 'biological_scientists', 'medical_scientists', 'chemist', 'scientist', 'materials_scientist', 'physical_scientists', 'lawyer', 'tutor', 'archivist', 'curator', 'director', 'scout', 'journalist', 'writers', 'dentists', 'surgical_technologist', 'orderlie', 'waiters', 'waitresse', 'cleaner', 'travel_guide', 'recreation_worker', 'cashier', 'retail_salesperson', 'baker', 'food_batchmaker', 'sales_worker', 'stockers_and_order_filler']

# 为了便于使用，选择每个类别的前10个职业
male_stereotype = ['facilities_manager', 'cost_estimator', 'information_security_analyst', 'database_administrator', 'programmer', 'software_developer', 'aerospace_engineer', 'chemical_engineer', 'civil_engineer', 'industrial_engineer']
female_stereotype = ['fundraising_manager', 'human_resources', 'psychologist', 'social_workers', 'searcher', 'librarians', 'interior_designer', 'actor', 'editor', 'interpreters']
neutral = ['chief_executive', 'manager', 'marketing_manager', 'sales_manager', 'financial_manager', 'purchasing_manager', 'compliance_officer', 'logistician', 'management_analyst', 'auditor']

# 合并所有职业
professions = male_stereotype + female_stereotype + neutral
gender_terms = ['he', 'she', 'man', 'woman']
