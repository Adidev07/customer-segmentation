# Model / system card

## Intended use

Discover customer groups and evaluate clustering stability using independent fits. Educational local experiment; no production deployment is implied.

## Data

Synthetic arrays, manually written documents or replayed JSON. No real user records. Seeds and sample inputs are recorded in app.py and sample_input.json.

## Method

Standardize features using training statistics. K-means++ chooses dispersed initial centers; Lloyd iterations alternate nearest-center assignment and mean updates. Empty centers retain their previous position.

## Evaluation protocol

Compare k=2,3,4,5 with a silhouette estimate on held-out rows. Report adjusted Rand index using synthetic labels solely as an external diagnostic, not for fitting.

## Known limits

Silhouette favors compact Euclidean clusters. Segment IDs have no inherent business meaning; synthetic separation makes the example easy.

## Verification status

Local tests and default experiment are executed during packaging. Live model generation is not exercised in the build. A successful validation result is limited to the checks implemented in code.

## Further work

Add mixed feature handling, bootstrap stability, segment profiles and a practical strategy for naming segments without using hidden labels.
