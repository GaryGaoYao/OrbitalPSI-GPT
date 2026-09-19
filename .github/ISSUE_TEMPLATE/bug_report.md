---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

name: 🐛 Bug Report
description: Report a reproducible problem with OrbitalPSI-GPT
title: "[Bug]: "
labels:
  - bug
body:
  - type: markdown
    attributes:
      value: |
        Thank you for helping improve OrbitalPSI-GPT.

        Please do not upload patient-identifiable data, clinical images,
        or other sensitive information.

  - type: dropdown
    id: component
    attributes:
      label: Component
      description: Which component is affected?
      options:
        - Desktop application
        - Segmentation / anatomy reconstruction
        - Landmark detection
        - Statistical shape model
        - Implant generation
        - Text-driven refinement
        - Model inference
        - Installation / dependencies
        - Documentation
        - Other
    validations:
      required: true

  - type: textarea
    id: problem
    attributes:
      label: Problem description
      description: Please describe what happened.
      placeholder: Describe the problem clearly and concisely.
    validations:
      required: true

  - type: textarea
    id: reproduce
    attributes:
      label: Steps to reproduce
      description: Please provide the minimum steps required to reproduce the issue.
      placeholder: |
        1. Start...
        2. Load...
        3. Run...
        4. Observe...
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: Expected behavior
      placeholder: What did you expect to happen?

  - type: input
    id: version
    attributes:
      label: OrbitalPSI-GPT version
      placeholder: e.g. software-v0.1.0
    validations:
      required: true

  - type: input
    id: system
    attributes:
      label: Operating system
      placeholder: e.g. Windows 11 / Ubuntu 24.04

  - type: textarea
    id: logs
    attributes:
      label: Logs or screenshots
      description: |
        Paste relevant logs or screenshots here.
        Please remove all patient-identifiable or sensitive information.

  - type: checkboxes
    id: checks
    attributes:
      label: Confirmation
      options:
        - label: I have checked that this issue has not already been reported.
          required: true
        - label: I have not included patient-identifiable or sensitive clinical information.
          required: true
