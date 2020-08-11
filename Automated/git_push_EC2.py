# -*- coding: utf-8 -*-


import codecs
import os
import sys

from git import Repo

sys.setdefaultencoding("utf-8")
# sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

os.chdir('/home/ec2-user/hectoramirez.github.io/covid')

PATH_OF_GIT_REPO = r'/home/ec2-user/hectoramirez.github.io/.git'

def git_push():

    repo = Repo(PATH_OF_GIT_REPO)
    repo.git.add('--all')  # update=True)
    repo.index.commit('daily')
    origin = repo.remote(name='origin')
    origin.push()


git_push()
