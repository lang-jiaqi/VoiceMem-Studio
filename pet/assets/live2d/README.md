# Live2D model package

Place a standard Cubism model3/moc3 folder here. Keep all paths referenced by the
model JSON relative and intact:

```text
avatar/
  avatar.model3.json
  avatar.moc3
  avatar.physics3.json       optional
  avatar.pose3.json          optional
  avatar.cdi3.json           optional
  textures/*.png
  motions/*.motion3.json     optional
  expressions/*.exp3.json   optional
```

The runtime also needs the official Cubism Core browser library at:

```text
vendor/live2dcubismcore.min.js
```

Cubism Core is not redistributed in this repository. Download it from the
official Cubism SDK after reviewing and accepting Live2D's license. A missing
Core or invalid model produces an explicit diagnostic; no legacy avatar is
substituted.
