"""
Keep pytest away from this folder: art_recipes.py needs Python >= 3.12 and the screamingface
package, neither of which the test environment has.
"""

collect_ignore = ["art_recipes.py"]
