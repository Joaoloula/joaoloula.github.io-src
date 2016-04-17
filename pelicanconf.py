#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'João Loula'
SITENAME = 'Waste Books'
TAGLINE = 'Collected musings on math and programming.'
SITEURL = 'http://joaoloula.github.io'
THEME = 'pure-single-master'
PATH = 'content'
STATIC_PATHS = ['images', 'pages', 'pdfs']
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
    ('github-square', 'https://github.com/joaoloula'),
    ('linkedin-square', 'https://fr.linkedin.com/in/joão-loula-2836b9107'),
    ('envelope-square', 'mailto:joao.campos-loula@polytechnique.edu'),
)
DISQUS_SITENAME = 'mydisqus'
DISQUS_ON_PAGESE = True
GOOGLE_ANALYTICS = 'UA-00000000-0'
COVER_IMG_URL = '/images/cover_stone.jpg'
PROFILE_IMG_URL = '/images/joao.jpg'
DEFAULT_PAGINATION = 10

# Plugins
PLUGIN_PATHS = ["plugins"]
PLUGINS = ['render-math', 'pelican-cite']

# Uncomment following line if you want document-relative URLs when developing
# RELATIVE_URLS = True
