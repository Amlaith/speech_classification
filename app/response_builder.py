import requests
from datetime import datetime, timedelta


def today():
    today_date = datetime.now()
    r = requests.get(f'https://ruz.fa.ru/api/schedule/group/110687?start={today_date.strftime("%Y.%m.%d")}&finish={today_date.strftime("%Y.%m.%d")}&lng=1')
    answer = f'<h2>Расписание на сегодня, {today_date.strftime("%Y.%m.%d")}:</h2>'
    for lesson in r.json():
        answer += (f"<p>{lesson['beginLesson']} - {lesson['endLesson']} \n  {lesson['building']}  \n  {lesson['auditorium']}  \n  {lesson['discipline']}  \n  {lesson['lecturer']} \n\n</p>")
    answer = '<div>' + answer + '</div>'
    return answer
    

def tomorrow():
    tmr_date = datetime.now() + timedelta(1)
    r = requests.get(f'https://ruz.fa.ru/api/schedule/group/110687?start={tmr_date.strftime("%Y.%m.%d")}&finish={tmr_date.strftime("%Y.%m.%d")}&lng=1')
    answer = f'<h2>Расписание на завтра, {tmr_date.strftime("%Y.%m.%d")}:</h2>'
    for lesson in r.json():
        answer += ''.join([
            f"{lesson['beginLesson']} - {lesson['endLesson']}", '\n  ',
            lesson['building'], '\n  ',
            lesson['auditorium'], '\n  ',
            lesson['discipline'], '\n  ',
            lesson['lecturer'], '\n',
            ])
        answer += '\n'
        answer = '<div>' + answer + '</div>'
    return answer

def three():
    answer = '<h2>Новости Университета:</h2>'
    return answer

def not_sure():
    answer = '<h2>Не удалось распознать команду</h2>'
    return answer

funcs = {
    'today': today,
    'tomorrow': tomorrow,
    'news': three,
    'not_sure': not_sure
}

def render_response(command):
    if command in funcs:
        return funcs[command]()
    else:
        return 'Command not found'

