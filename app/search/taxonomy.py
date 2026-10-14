"""A small taxonomy of wealth-management / KYC document types used for query expansion.

Why: small embedding models do not know domain equivalences such as "a utility bill is a proof of
address" (see README, "Design decisions"). The query is matched against each type's phrases with
the embedding model. When it clearly refers to a type, documents of that type are retrieved
(a) lexically, by the type's phrases, and (b) semantically: the query vector is blended with the
type's centroid before the nearest-chunk search.
"""

from dataclasses import dataclass

import numpy as np

from app.indexing.embedder import Embedder, l2_normalize


@dataclass(frozen=True)
class DocumentType:
    key: str
    label: str
    phrases: tuple[str, ...]

    @property
    def all_phrases(self) -> tuple[str, ...]:
        return (self.label, *self.phrases)


DOCUMENT_TYPES: tuple[DocumentType, ...] = (
    DocumentType(
        "proof_of_address",
        "Proof of address",
        (
            "address proof", "proof of residence", "address verification", "utility bill",
            "electricity bill", "gas bill", "water bill", "council tax bill",
            "tenancy agreement", "lease agreement",
        ),
    ),
    DocumentType(
        "identity_document",
        "Identity document",
        (
            "proof of identity", "photo ID", "passport", "driver license", "driving licence",
            "national identity card", "ID card",
        ),
    ),
    DocumentType(
        "proof_of_income",
        "Proof of income",
        ("income verification", "payslip", "pay stub", "salary slip", "employment letter", "W-2"),
    ),
    DocumentType(
        "tax_document",
        "Tax document",
        ("tax return", "Form 1040", "Form 1099", "tax assessment"),
    ),
    DocumentType(
        "bank_statement",
        "Bank statement",
        ("account statement", "checking account", "savings account", "current account",
         "wire transfer"),
    ),
    DocumentType(
        "investment_statement",
        "Investment statement",
        ("brokerage statement", "portfolio statement", "custody statement",
         "securities holdings", "stocks and ETFs"),
    ),
    DocumentType(
        "risk_profile",
        "Risk profile",
        ("risk tolerance questionnaire", "risk appetite", "investor profile",
         "suitability assessment", "risk score"),
    ),
    DocumentType(
        "estate_planning",
        "Estate planning document",
        ("last will and testament", "trust deed", "trust agreement", "power of attorney",
         "beneficiary designation"),
    ),
    DocumentType(
        "loan_or_mortgage",
        "Loan or mortgage",
        ("mortgage statement", "home loan", "loan agreement", "credit facility",
         "outstanding principal"),
    ),
    DocumentType(
        "insurance_policy",
        "Insurance policy",
        ("life insurance", "life cover", "policy schedule", "insurance premium", "death benefit"),
    ),
    DocumentType(
        "source_of_wealth",
        "Source of wealth",
        ("source of funds", "AML declaration", "origin of funds", "proceeds from business sale"),
    ),
    DocumentType(
        "investment_policy",
        "Investment policy statement",
        ("strategic asset allocation", "rebalancing policy", "investment mandate"),
    ),
    DocumentType(
        "meeting_notes",
        "Meeting notes",
        ("client meeting", "call notes", "review meeting", "action items"),
    ),
)  # fmt: skip


@dataclass(frozen=True)
class TypeMatch:
    type: DocumentType
    similarity: float


class TaxonomyMatcher:
    def __init__(self, embedder: Embedder, types: tuple[DocumentType, ...] = DOCUMENT_TYPES):
        self.types = types
        self._owners: list[int] = []
        phrases: list[str] = []
        for index, doc_type in enumerate(types):
            for phrase in doc_type.all_phrases:
                self._owners.append(index)
                phrases.append(phrase)
        self._owners_arr = np.array(self._owners)
        self._vectors = embedder.embed_documents(phrases)
        # The centroid of a type's phrases stands for "a document of this type" in vector space.
        self._centroids = {
            doc_type.key: l2_normalize(self._vectors[self._owners_arr == index].mean(axis=0))
            for index, doc_type in enumerate(types)
        }

    def centroid(self, doc_type: DocumentType) -> np.ndarray:
        return self._centroids[doc_type.key]

    def match(
        self,
        query_vector: np.ndarray,
        min_similarity: float,
        top_k: int = 2,
        margin: float = 0.2,
    ) -> list[TypeMatch]:
        """Types whose closest phrase is at least `min_similarity` to the query, best first.

        A runner-up type is kept only if it is nearly as close as the best one ("address proof"
        names proof of address, not proof of income).
        """
        similarities = self._vectors @ query_vector
        best = np.full(len(self.types), -1.0)
        np.maximum.at(best, self._owners_arr, similarities)
        order = np.argsort(-best)[:top_k]
        cutoff = max(min_similarity, float(best[order[0]]) - margin)
        return [TypeMatch(self.types[i], float(best[i])) for i in order if best[i] >= cutoff]

    def blend(
        self, query_vector: np.ndarray, matches: list[TypeMatch], weight: float
    ) -> np.ndarray:
        """The query vector pulled towards the centroids of the matched types.

        `normalize(query + weight * sum(similarity * centroid))`: "address proof" then points at
        "a typical proof-of-address document", next to an electricity bill. The more confident the
        type match, the stronger the pull; with no match the query vector is returned unchanged.
        """
        if not matches:
            return query_vector
        pull = sum(m.similarity * self.centroid(m.type) for m in matches)
        return l2_normalize(query_vector + weight * pull)
