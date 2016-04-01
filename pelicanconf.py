#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'João Loula'
SITENAME = 'João Loula'
SITEURL = ''
THEME = 'pure-single-master'
PATH = 'content'

TIMEZONE = 'Europe/Paris'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Social widget
SOCIAL = (
    ('github-alt', 'https://github.com/joaoloula'),
    ('linkedin', 'https://fr.linkedin.com/in/joão-loula-2836b9107'),
	('envelope', 'mailto:joao.campos-loula@polytechnique.edu')
)
DISQUS_SITENAME = 'mydisqus'
DISQUS_ON_PAGESE = True
GOOGLE_ANALYTICS = 'UA-00000000-0'
COVER_IMG_URL = '/home/loula/Pictures/Wallpapers/reading_mountains.jpg'
PROFILE_IMG_URL = '/home/loula/Pictures/Joao/profilepic_june.jpg'
SOCIAL = (('You can add links in your config file', '#'),
          ('Another social link', '#'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
