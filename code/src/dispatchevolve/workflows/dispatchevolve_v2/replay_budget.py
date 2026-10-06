"""Process-shared replay cost ledger, measured in evolution-set row equivalents."""
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
import sqlite3


class ReplayBudgetExhausted(RuntimeError):
    pass


class ReplayBudget:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=60)
        try:
            connection.execute('BEGIN IMMEDIATE')
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self, rows, limit):
        if rows <= 0:
            raise ValueError('replay budget requires a nonempty evolution set')
        capacity = int(Decimal(str(limit)) * rows)
        with self.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS budget (rows INTEGER, capacity INTEGER, used INTEGER, denied INTEGER)')
            db.execute('CREATE TABLE IF NOT EXISTS charges (id INTEGER PRIMARY KEY, evaluation_key TEXT, rows INTEGER)')
            saved = db.execute('SELECT rows, capacity FROM budget').fetchone()
            if saved is None:
                db.execute('INSERT INTO budget VALUES (?, ?, 0, 0)', (rows, capacity))
            elif saved != (rows, capacity):
                raise ValueError('replay budget configuration differs from the persisted ledger')

    def reserve(self, rows, evaluation_key):
        if rows <= 0:
            raise ValueError('replay charge must contain positive rows')
        denied = False
        with self.transaction() as db:
            capacity, used = db.execute('SELECT capacity, used FROM budget').fetchone()
            if used + rows > capacity:
                db.execute('UPDATE budget SET denied = denied + 1')
                denied = True
            else:
                db.execute('UPDATE budget SET used = used + ?', (rows,))
                db.execute('INSERT INTO charges(evaluation_key, rows) VALUES (?, ?)', (evaluation_key, rows))
        if denied:
            raise ReplayBudgetExhausted('insufficient evolution replay budget')

    def snapshot(self):
        with self.transaction() as db:
            rows, capacity, used, denied = db.execute('SELECT rows, capacity, used, denied FROM budget').fetchone()
            calls = db.execute('SELECT COUNT(*) FROM charges').fetchone()[0]
        return {'unit': 'full_evolution_replay', 'evolution_rows': rows,
                'limit': capacity / rows, 'used': used / rows,
                'remaining': (capacity - used) / rows, 'charged_rows': used,
                'charged_calls': calls, 'denied_calls': denied,
                'test_evaluations_included': False}

    def check_stopped(self):
        snapshot = self.snapshot()
        if snapshot['denied_calls'] or snapshot['remaining'] == 0:
            raise ReplayBudgetExhausted('evolution replay budget exhausted')
