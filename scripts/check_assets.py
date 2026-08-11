"""Print the readiness of each optional application capability."""

import json

from backend.services import capability_status


def main():
    capabilities = capability_status()
    print(json.dumps(capabilities, indent=2))
    return 0 if all(item["available"] for item in capabilities.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
