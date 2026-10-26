# ART as ScreamingFace recipes

`art_recipes.py` expresses one round of ART as two [ScreamingFace](https://docs.screamingface.ai)
recipes: `art_one_judge` (best-of-N by one judge's scores) and `art_peer_panel` (every model
judges, then a plurality vote). It needs Python >= 3.12 and the `screamingface` package, which
is not a dependency of this repository. Both recipes compile with the ScreamingFace client
(checked against screamingface commit 2b04cde).
