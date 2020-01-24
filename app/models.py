from sqlalchemy import Integer, ForeignKey, String, Column, Float, Boolean
from datetime import datetime
from app import db  


class ConfusionMatrix(db.Model):
    id = Column(Integer, primary_key=True, autoincrement=True)
    total = Column(Integer, default=0)
    true_positives = Column(Integer, default=0)
    true_negatives = Column(Integer, default=0)
    false_positives = Column(Integer, default=0)
    false_negatives = Column(Integer, default=0)

    def add_prediction(self, actual, predicted):
        if actual == "correct":
            if predicted == "positive":
                self.true_positives += 1
            else:
                self.true_negatives += 1
        else:
            if predicted == "positive":
                self.false_positives += 1
            else:
                self.false_negatives += 1
        self.total += 1
        db.session.commit()
    
    def get_matrix(self):
        return {
            'total': self.total,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives
        }