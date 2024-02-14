from functools import cached_property


class UI_Styles:
    @cached_property
    def app_styles(self):
        return '''
        #label_the_word { font-size: xx-large; color: gold; font-family: "Monospace" ; text-align: center; }
        '''

    def __str__(self):
        return UI_Styles.app_styles
