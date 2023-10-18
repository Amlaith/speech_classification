import requests
from datetime import datetime, timedelta


def today():
    today_date = datetime.now()
    r = requests.get(f'https://ruz.fa.ru/api/schedule/group/110687?start={today_date.strftime("%Y.%m.%d")}&finish={today_date.strftime("%Y.%m.%d")}&lng=1')
    answer = ''
    for lesson in r.json():
        answer += ''.join([
            f"{lesson['beginLesson']} - {lesson['endLesson']}", '\n',
            lesson['building'], '\n',
            lesson['auditorium'], '\n',
            lesson['discipline'], '\n',
            lesson['lecturer'], '\n',
            ])
        answer += '\n'
    return answer
    

def tomorrow():
    tmr_date = datetime.now() + timedelta(1)
    r = requests.get(f'https://ruz.fa.ru/api/schedule/group/110687?start={tmr_date.strftime("%Y.%m.%d")}&finish={tmr_date.strftime("%Y.%m.%d")}&lng=1')
    answer = ''
    for lesson in r.json():
        answer += ''.join([
            f"{lesson['beginLesson']} - {lesson['endLesson']}", '\n',
            lesson['building'], '\n',
            lesson['auditorium'], '\n',
            lesson['discipline'], '\n',
            lesson['lecturer'], '\n',
            ])
        answer += '\n'
    return answer

def three():
    return 'That\'s three'

funcs = {
    'сегодня': today,
    'завтра': tomorrow,
    'three': three
}

def render_answer(command):
    if command in funcs:
        return funcs[command]()
    else:
        return 'Command not found'

