from datetime import datetime


DATE_FORMAT = '%Y-%m-%d'


def create_date(date):
    return datetime.strptime(date, DATE_FORMAT)