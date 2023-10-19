def decode_command(command):
    print(type(command)==str)
    if type(command) == str:
        return command.lower()
    else:
        return None