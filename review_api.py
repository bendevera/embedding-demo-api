from app import app, db
from app.models import ConfusionMatrix

@app.shell_context_processor
def make_shell_context():
    return {'app': app, 'db': db, 'ConfusionMatrix': ConfusionMatrix}