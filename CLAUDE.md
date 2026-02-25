# Project Notes

## Training
- Training is done via SSH node.

## Definition of Done
A task is not considered done until:
1. Tested in the testing suite to verify correctness.
2. Added more specific ad-hoc tests for complete specificity.

## Testing Guidelines
- No more than 1 test per function.
- Each test must complete in under 10 seconds.
- Mocking is OK and preferred for external dependencies (physics, rendering, file I/O).
