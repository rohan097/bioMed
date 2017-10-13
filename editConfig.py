import json
import os
import re


def check_file():
    if os.path.isfile('config.json'):
        return True
    else:
        return False


def edit():
    with open('config.json', 'r') as f:
        config = json.load(f)
    print(config)


def write(obj):
    with open('config.json', 'w') as f:
        json.dump(obj, f)
    print("File  written!")


def create():
    keys = input("Enter the parameters, separated by a space: \n").strip().split(' ')
    print(keys)
    data = {}
    for i in keys:
        data[i] = input("Enter the value to be associated with %s: " % i)
        if re.match('/*[A-Za-z]+', data[i]):
            pass
        else:
            data[i] = float(data[i])

    print("The config file to be generated is: \n", data)
    choice = input("Do you wish to write it to file? [Y/N]: ")
    if choice in 'Yy':
        write(data)
    else:
        parameter = input("Enter the parameter you wish to edit: \n").strip().split(' ')
        for i in parameter:
            data[i] = input("Enter the new value for %s" % i)
        write(data)


def main():
    file_present = check_file()
    if file_present:
        edit()
    else:
        create()


if __name__ == '__main__':
    main()
