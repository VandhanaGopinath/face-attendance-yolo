# LFW face-verification results

- Test pairs: 1000 (500 same-person, 500 different-person), 1549 photos
- Threshold: 0.225 (chosen on 2200 train pairs, train accuracy 99.82%)
- **Verification accuracy on the test pairs: 99.30%**
- Same-person pairs accepted: 99.00%   |   different-person pairs wrongly accepted: 0.40%
- ROC-AUC: 0.9977
- Accuracy at the app's default threshold (0.4): 98.20%
- Photos where YOLO found no face and the central-crop fallback was used: 0 of 1549

Protocol: official LFW dev-train pairs choose the threshold, official dev-test pairs measure accuracy.
