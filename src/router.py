"""
Stage 2: Simple/Complex Query Router
Naive Bayes classifier on TF-IDF features of the original user query.
Trained independently of Stage 1 — classifies query complexity
before any PII processing occurs.
"""

import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score

# ── Dataset ───────────────────────────────────────────────────────────
# Hand-written — no public dataset relabeled because none maps cleanly
# to the simple/complex distinction needed here

SIMPLE_QUERIES = [
    "What is the company's refund policy?",
    "Who is the HR contact?",
    "What is two-factor authentication?",
    "When does the office open?",
    "What is Rahul's job title?",
    "Who manages the Mumbai office?",
    "Define API key rotation.",
    "What is the leave policy?",
    "What is an Aadhaar number?",
    "What is a PAN card?",
    "Who is the team lead?",
    "What is the company email?",
    "What does RAG stand for?",
    "What is a vector database?",
    "Who is the CEO?",
    "What is the office address?",
    "What is HTTP?",
    "What is encryption?",
    "What is a firewall?",
    "What is MFA?",
    "Who handles payroll?",
    "What is the onboarding process?",
    "What is a webhook?",
    "What is an API?",
    "What is a database?",
    "Who is the SPOC for this project?",
    "What is the ticket number?",
    "When was the policy last updated?",
    "What is the SLA?",
    "What is a VPN?",
    "What is TLS?",
    "What is OAuth?",
    "What is a microservice?",
    "What is Docker?",
    "What is Kubernetes?",
    "What is CI/CD?",
    "What is a pull request?",
    "What is a bug?",
    "Who filed the complaint?",
    "What is the project deadline?",
    "What is the vendor's contact number?",
    "What is the invoice amount?",
    "When is the next sprint?",
    "What is the status of the ticket?",
    "Who approved the request?",
    "What is the password policy?",
    "What is the data retention period?",
    "What is GDPR?",
    "What is the DPDP Act?",
    "What is a data breach?",
    "Who is the data protection officer?",
    "What is a cookie?",
    "What is an IP address?",
    "What is a server?",
    "What is the cloud?",
    "What is AWS?",
    "What is S3?",
    "What is a subnet?",
    "What is a load balancer?",
    "What is the difference between HTTP and HTTPS?",
    "What is a hash function?",
    "What is base64?",
    "What is JSON?",
    "What is XML?",
    "What is REST?",
    "What is GraphQL?",
    "What is a token?",
    "What is a session?",
    "What is a cookie?",
    "What is a cache?",
    "What is a proxy?",
    "What is DNS?",
    "What is SSL?",
    "What is a certificate?",
    "What is a public key?",
    "What is a private key?",
    "What is symmetric encryption?",
    "What is a SHA256 hash?",
    "What is the OSI model?",
    "What is TCP/IP?",
    "What is a port number?",
    "What is localhost?",
    "What is a runtime error?",
    "What is a null pointer?",
    "What is recursion?",
    "What is a loop?",
    "What is a function?",
    "What is a variable?",
    "What is Python?",
    "What is Java?",
    "What is SQL?",
    "What is a join in SQL?",
    "What is an index?",
    "What is a primary key?",
    "What is a foreign key?",
    "What is normalization?",
    "What is a transaction?",
    "What is ACID?",
    "What is a schema?",
    "What is a stored procedure?",
    "What is a trigger?",
    "What is a view in SQL?",
    "What is NoSQL?",
    "What is MongoDB?",
    "What is Redis?",
]

COMPLEX_QUERIES = [
    "Compare Rahul and Priya's performance over the last two quarters and recommend corrective action.",
    "Summarize the implications of the new data policy on regional operations across all three zones.",
    "Why did the Mumbai team miss Q3 targets and what should be done differently next quarter?",
    "Analyze the relationship between API key exposure and the July security incident.",
    "What are the tradeoffs between storing credentials in environment variables versus a secrets manager?",
    "Explain the step by step process for onboarding a new vendor including all compliance requirements.",
    "How does the DPDP Act affect our current data retention and processing policies?",
    "Compare the performance of the staging and production environments over the last month.",
    "What corrective actions should be taken following the findings in the Q3 audit report?",
    "Evaluate the risk of moving our authentication system from OAuth to SAML.",
    "Summarize all open compliance violations and recommend a remediation priority order.",
    "How should the company restructure its data pipeline to comply with RBI data localisation rules?",
    "Analyze the security incident from last week and identify all affected systems.",
    "What are the architectural differences between our current monolith and the proposed microservices design?",
    "Compare the cost and performance implications of running on AWS versus GCP for our workload.",
    "How does our current backup strategy hold up against the RPO and RTO requirements in the SLA?",
    "Explain why the authentication failures increased after the last deployment and how to fix them.",
    "What is the long-term roadmap for migrating our legacy database to a cloud-native solution?",
    "Summarize the vendor evaluation results and recommend which vendor to select and why.",
    "How should we prioritize the open security findings given our current engineering bandwidth?",
    "Compare the precision and recall tradeoffs of the three candidate ML models for fraud detection.",
    "What are the regulatory implications of storing customer Aadhaar data in a third-party cloud?",
    "Analyze the root cause of the production outage last Tuesday and propose preventive measures.",
    "How does the proposed API gateway design affect latency across our current service topology?",
    "Evaluate the effectiveness of our current incident response process based on last quarter's incidents.",
    "What changes to our CI/CD pipeline would reduce deployment failures by the most?",
    "Compare the data governance maturity of our analytics and engineering teams.",
    "How should we handle the conflict between data minimisation requirements and our ML training needs?",
    "Summarize the findings from the penetration test and prioritize remediation steps.",
    "What is the impact of switching from a monolithic to an event-driven architecture on our team?",
    "Analyze how the recent RBI circular affects our payment processing flow end to end.",
    "How do we balance the need for audit logging with our data retention and privacy obligations?",
    "What are the tradeoffs between using a managed Kubernetes service versus self-hosted?",
    "Compare our current SLO performance against industry benchmarks and identify gaps.",
    "How should we restructure our team to better support the new microservices architecture?",
    "Explain the cascading effects of the database connection pool exhaustion last Friday.",
    "What are the privacy risks of our current RAG implementation and how should we mitigate them?",
    "How does our data classification policy interact with our third-party vendor agreements?",
    "Summarize the business impact of the API downtime last month across all dependent services.",
    "What is the best approach for gradually migrating our authentication system with zero downtime?",
    "Compare the security posture of our three cloud environments and recommend improvements.",
    "How should we handle data subject access requests at scale given our current architecture?",
    "Analyze the performance degradation pattern in our recommendation engine over the last 90 days.",
    "What are the compliance gaps between our current practices and ISO 27001 requirements?",
    "How do we design a rollback strategy for the upcoming database migration?",
    "Summarize the key risks identified in the vendor due diligence report for the new payment processor.",
    "What is the most effective way to reduce our cloud spend without impacting service reliability?",
    "How should we approach zero-trust implementation given our current network topology?",
    "Analyze the data flow between our CRM and ERP systems and identify integration gaps.",
    "What are the long-term implications of our current technical debt on system reliability?",
    "How do we ensure consistent data quality across our distributed microservices?",
    "Compare the regulatory requirements for data handling under GDPR versus the DPDP Act.",
    "What is the best strategy for handling API versioning across 20 dependent internal services?",
    "How should we design our disaster recovery plan to meet a 4-hour RTO requirement?",
    "Analyze the trade-offs between eventual consistency and strong consistency for our use case.",
    "What are the security implications of our current service mesh configuration?",
    "How do we build a data lineage system that covers our entire analytics pipeline?",
    "Summarize the competitive landscape for PII detection tools and compare against our approach.",
    "What is the impact of increasing our model inference batch size on latency and throughput?",
    "How should we handle model drift detection for our production recommendation system?",
    "Analyze the fairness implications of our credit scoring model across demographic groups.",
    "What is the optimal feature store architecture for our real-time and batch ML pipelines?",
    "How do we implement canary deployments for ML models without impacting user experience?",
    "Compare the MLOps maturity of our team against industry best practices.",
    "What are the tradeoffs between fine-tuning a small model versus prompting a large model for our task?",
    "How should we design our data labeling pipeline to minimize annotation bias?",
    "Analyze the cost-benefit tradeoff of building versus buying a document intelligence solution.",
    "What governance framework should we adopt for responsible AI deployment in our products?",
    "How do we ensure reproducibility of our ML experiments across different infrastructure environments?",
    "Summarize the key architectural decisions made in the last six months and their outcomes.",
    "What are the scaling bottlenecks in our current data ingestion pipeline?",
    "How should we approach multi-tenancy isolation in our SaaS product architecture?",
    "Analyze the latency breakdown across our entire request path and identify optimization opportunities.",
    "What is the most robust approach for handling partial failures in our distributed transaction system?",
    "How do we design an effective alerting strategy that minimizes alert fatigue?",
    "Compare the observability coverage of our services and identify monitoring gaps.",
    "What are the implications of our current logging strategy for GDPR compliance?",
    "How should we handle breaking changes in our public API without disrupting partners?",
    "Analyze the dependency graph of our services and identify single points of failure.",
    "What is the best approach for migrating our on-premise data warehouse to the cloud?",
    "How do we design a cost allocation model for shared infrastructure across business units?",
    "Summarize the outcomes of the last three architecture review board meetings.",
    "What are the risks and mitigations for adopting a new ORM framework in production?",
    "How should we structure our engineering teams to minimize cross-team dependencies?",
    "Analyze the build time trends across our repositories and recommend optimizations.",
    "What is the most effective way to implement progressive rollouts for database schema changes?",
    "How do we ensure our API rate limiting strategy is fair and effective across all client types?",
    "Compare the performance characteristics of our three caching strategies under peak load.",
    "What are the security risks of our current secrets rotation process and how to improve it?",
    "How should we approach capacity planning for the upcoming peak traffic season?",
    "Analyze the correlation between deployment frequency and incident rate over the last year.",
    "What is the most effective way to reduce MTTR for production incidents in our team?",
    "How do we build a self-service data platform that maintains governance and security?",
    "Summarize the lessons learned from the last major incident and recommend process changes.",
    "What are the tradeoffs between synchronous and asynchronous communication in our architecture?",
    "How should we design our data mesh to support domain ownership while ensuring interoperability?",
    "Analyze the impact of our current technical debt on new feature delivery velocity.",
    "What is the best strategy for consolidating our fragmented authentication systems?",
    "How do we implement effective chaos engineering without impacting production reliability?",
]

# Combine and label
queries = SIMPLE_QUERIES + COMPLEX_QUERIES
labels  = ["simple"] * len(SIMPLE_QUERIES) + ["complex"] * len(COMPLEX_QUERIES)

print(f"Dataset: {len(SIMPLE_QUERIES)} simple, {len(COMPLEX_QUERIES)} complex")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    queries, labels, test_size=0.2, random_state=42, stratify=labels
)

# Pipeline: TF-IDF + Naive Bayes
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams and bigrams
        min_df=1,
        max_features=5000,
        sublinear_tf=True,    # log normalization
    )),
    ("clf", MultinomialNB(alpha=0.1)),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

precision = precision_score(y_test, y_pred, pos_label="complex")
recall    = recall_score(y_test, y_pred, pos_label="complex")
print(f"Precision (complex): {precision:.3f}")
print(f"Recall    (complex): {recall:.3f}")

# Test on new examples
test_queries = [
    ("What is the leave policy?", "simple"),
    ("Who is the HR contact?", "simple"),
    ("Compare Rahul and Priya's performance and recommend action.", "complex"),
    ("Analyze the security incident and identify all affected systems.", "complex"),
    ("What is MFA?", "simple"),
    ("How should we restructure the data pipeline for DPDP compliance?", "complex"),
]

print("\nPredictions on new examples:")
for query, expected in test_queries:
    predicted = pipeline.predict([query])[0]
    status    = "✓" if predicted == expected else "✗"
    print(f"  {status} [{predicted}] {query[:70]}")

# Save model
import os
os.makedirs("models", exist_ok=True)
with open("models/router_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\nModel saved to models/router_pipeline.pkl")