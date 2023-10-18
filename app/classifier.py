def decode_command(command):
    if type(command) == 'str':
        return command.lower()
    else:
        return None