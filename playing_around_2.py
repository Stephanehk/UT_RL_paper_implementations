

_current_index = 2
_accum_meters = [1,2,3,4,5,6]
stuff = 100.0 * float(_accum_meters[_current_index]) \
                    / float(_accum_meters[-1])
print (stuff)
