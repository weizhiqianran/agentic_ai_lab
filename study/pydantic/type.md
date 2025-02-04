# An Introduction to Pydantic Types: Validation, Error Handling, and Real-world Examples

Pydantic is a powerful Python library that leverages type hints to validate, parse, and serialize data effortlessly. In this article, we introduce several key Pydantic types, explain their uses, and walk through **seven complete examples** that demonstrate real-world usage with robust error handling. The examples have been grouped into three categories to help you quickly find the type of validation you’re interested in.

---

## Table of Contents

1. [Introduction to Pydantic Types](#introduction-to-pydantic-types)
2. [Sensitive Data and Payment Models](#sensitive-data-and-payment-models)
   - [Payment Transaction Model](#payment-transaction-model)
   - [Membership Model](#membership-model)
   - [User Contact Information](#user-contact-information)
3. [File System and Encoding Models](#file-system-and-encoding-models)
   - [File System Paths Model](#file-system-paths-model)
   - [Encoded Data Model](#encoded-data-model)
4. [Network and Connection Models](#network-and-connection-models)
   - [External Endpoints Configuration](#external-endpoints-configuration)
   - [Service Connection DSNs](#service-connection-dsns)
   - [Network Settings](#network-settings)
5. [Conclusion](#conclusion)

---

## Introduction to Pydantic Types

Pydantic uses Python’s type annotations to define the structure and constraints of your data. This allows you to create robust models that automatically validate inputs, reducing bugs and enforcing consistency in your code. Whether you need to ensure a number is negative, a URL is well-formed, or a datetime value is in the past or future, Pydantic helps you achieve that with minimal code.

### Why Use Pydantic?

- **Automatic Data Validation:** Automatically check the type and constraints of each field.
- **Error Handling:** Detailed `ValidationError` exceptions help pinpoint the source of errors.
- **Serialization & Parsing:** Easily convert data to and from various formats.
- **Readability & Maintainability:** Models serve as clear, self-documenting code.

---

## Sensitive Data and Payment Models

These examples show how Pydantic handles sensitive information and user-related data validations.

### Payment Transaction Model

Pydantic provides specialized types for handling sensitive data, validating payment card numbers, and enforcing date constraints. In this example, we use the following types:

- **`SecretStr`**: Securely stores sensitive strings such as passwords.
- **`PaymentCardNumber`**: Validates and masks credit card numbers.
- **`PastDatetime`**: Ensures that the transaction date is in the past.
- **`FutureDatetime`**: Ensures that the card’s expiration date is in the future.

The code below demonstrates a Payment Transaction Model with robust error handling.

```python
from pydantic import BaseModel, SecretStr, PaymentCardNumber, PastDatetime, FutureDatetime, ValidationError
from datetime import datetime, timedelta

class PaymentTransaction(BaseModel):
    secret: SecretStr
    card_number: PaymentCardNumber
    transaction_date: PastDatetime
    card_expiration: FutureDatetime

# Create example dates:
yesterday = datetime.now() - timedelta(days=1)
one_year_from_now = datetime.now() + timedelta(days=365)

# --- Correct Usage ---
try:
    payment = PaymentTransaction(
        secret="supersecretpassword",
        card_number="4111111111111111",
        transaction_date=yesterday,
        card_expiration=one_year_from_now
    )
    print("Payment Transaction Model (Valid):")
    print(payment)
    # Access sensitive data and masked card number:
    print("Unmasked Secret:", payment.secret.get_secret_value())
    print("Masked Card Number:", payment.card_number.masked)
except ValidationError as e:
    print("Validation error in PaymentTransaction (Valid Data Test):")
    print(e.json())

# --- Error Handling Example ---
# This example uses a future transaction date, which is invalid for PastDatetime.
try:
    payment_invalid = PaymentTransaction(
        secret="supersecretpassword",
        card_number="4111111111111111",
        transaction_date=datetime.now() + timedelta(days=1),  # future date, should trigger error
        card_expiration=one_year_from_now
    )
except ValidationError as e:
    print("\nValidation error for invalid PaymentTransaction:")
    print(e.json())
```

**Key Points:**
- **`SecretStr`** masks the sensitive value when the model is printed.
- **`PaymentCardNumber`** validates the card number and provides a masked version for secure display.
- **`PastDatetime`** ensures that the `transaction_date` is a past date.
- **`FutureDatetime`** ensures that the `card_expiration` is set to a future date.
- The use of `try/except` blocks demonstrates robust error handling in case of invalid input data.

---

### Membership Model

This additional example demonstrates the use of `PastDatetime` and `FutureDatetime` to enforce constraints on membership dates. The model ensures that a member's start date is in the past and the membership end date is in the future.

```python
from pydantic import BaseModel, PastDatetime, FutureDatetime, ValidationError
from datetime import datetime, timedelta

class Membership(BaseModel):
    member_name: str
    membership_start: PastDatetime
    membership_end: FutureDatetime

# --- Correct Usage ---
try:
    membership = Membership(
        member_name="Alice",
        membership_start=datetime.now() - timedelta(days=30),  # 30 days ago
        membership_end=datetime.now() + timedelta(days=365)    # 1 year from now
    )
    print("\nMembership Model (Valid):")
    print(membership)
except ValidationError as e:
    print("Validation error in Membership (Valid Data Test):")
    print(e.json())

# --- Error Handling Example ---
# This example uses a start date in the future, which is invalid for PastDatetime.
try:
    membership_invalid = Membership(
        member_name="Bob",
        membership_start=datetime.now() + timedelta(days=1),   # future date, should trigger error
        membership_end=datetime.now() + timedelta(days=365)
    )
except ValidationError as e:
    print("\nValidation error for invalid Membership (start date in the future):")
    print(e.json())
```

**Key Points:**
- **`PastDatetime`** in the Membership model ensures that the `membership_start` date is in the past.
- **`FutureDatetime`** guarantees that the `membership_end` date is set to a future date.
- Robust error handling using `try/except` blocks helps catch and display validation errors, ensuring data integrity.

---

### User Contact Information

This example leverages:

- **`EmailStr`** for standard email validation.
- **`NameEmail`** for representing a user's contact information in the format `"Full Name <email@example.com>"`.

```python
from pydantic import BaseModel, EmailStr, NameEmail, ValidationError

class UserContact(BaseModel):
    email: EmailStr
    display_contact: NameEmail

# Correct usage
try:
    user = UserContact(
        email="john.doe@example.com",
        display_contact="John Doe <john.doe@example.com>"
    )
    print("\nUser Contact (Valid):")
    print(user)
except ValidationError as e:
    print("Validation error in UserContact:")
    print(e.json())

# Example with invalid email in display_contact (missing angle brackets)
try:
    user_invalid = UserContact(
        email="john.doe@example.com",
        display_contact="John Doe john.doe@example.com"  # invalid format
    )
except ValidationError as e:
    print("\nValidation error for invalid UserContact:")
    print(e.json())
```

---

## File System and Encoding Models

These examples focus on validations involving file system paths and encoded data.

### File System Paths Model

This example demonstrates using path-related types:

- **`FilePath`** ensures a given file path exists.
- **`DirectoryPath`** ensures a given directory exists.
- **`SocketPath`** validates that the given string represents a valid socket path.

> **Note:** Adjust the paths to ones that exist (or intentionally don’t) on your system to test the error handling.

```python
from pydantic import BaseModel, FilePath, DirectoryPath, SocketPath, ValidationError

class FileSystemPaths(BaseModel):
    file_path: FilePath
    directory_path: DirectoryPath
    socket_path: SocketPath

# Correct usage (ensure these paths exist on your system)
try:
    fs_paths = FileSystemPaths(
        file_path="/tmp/example_file.txt",         # adjust as needed
        directory_path="/tmp",                     # adjust as needed
        socket_path="/tmp/example_socket"          # adjust as needed or create a dummy socket file
    )
    print("\nFile System Paths Model (Valid):")
    print(fs_paths)
except ValidationError as e:
    print("Validation error in FileSystemPaths:")
    print(e.json())

# Example with invalid file path (non-existent file)
try:
    fs_paths_invalid = FileSystemPaths(
        file_path="/path/does/not/exist.txt",      # likely invalid
        directory_path="/tmp",
        socket_path="/tmp/example_socket"
    )
except ValidationError as e:
    print("\nValidation error for invalid FileSystemPaths:")
    print(e.json())
```

---

### Encoded Data Model

This example uses:

- **`Base64Bytes`** to work with base64-encoded bytes.
- We simulate **`Base64UrlEncoder`** by using Python's built-in URL-safe encoding with the `base64` module.

```python
from pydantic import BaseModel, Base64Bytes, ValidationError
import base64

class EncodedDataModel(BaseModel):
    data: Base64Bytes

    @property
    def url_safe_encoded(self) -> str:
        """
        Returns a URL-safe Base64-encoded version of the underlying data.
        """
        return base64.urlsafe_b64encode(self.data).decode()

# Correct usage
try:
    original_bytes = b"Hello, world!"
    encoded_bytes = base64.b64encode(original_bytes)
    encoded_model = EncodedDataModel(data=encoded_bytes)
    print("\nEncoded Data Model (Valid):")
    print("Base64 Encoded Data:", encoded_model.data)
    print("URL-Safe Encoded Data:", encoded_model.url_safe_encoded)
except ValidationError as e:
    print("Validation error in EncodedDataModel:")
    print(e.json())

# Example with invalid base64 data (not a valid base64-encoded bytes string)
try:
    invalid_encoded_model = EncodedDataModel(data=b"NotBase64!!")
except ValidationError as e:
    print("\nValidation error for invalid EncodedDataModel:")
    print(e.json())
```

---

## Network and Connection Models

These examples demonstrate how Pydantic validates network endpoints, connection strings, and IP addresses.

### External Endpoints Configuration

This model uses:

- **`AnyUrl`** for generic URL validation.
- **`WebsocketUrl`** for ensuring that a WebSocket URL is properly formatted.

```python
from pydantic import BaseModel, AnyUrl, WebsocketUrl, ValidationError

class EndpointsConfig(BaseModel):
    service_url: AnyUrl
    realtime_feed: WebsocketUrl

# Correct usage
try:
    config = EndpointsConfig(
        service_url="https://api.example.com/resource",
        realtime_feed="wss://feed.example.com/socket"
    )
    print("\nEndpoints Config (Valid):")
    print(config)
except ValidationError as e:
    print("Validation error in EndpointsConfig:")
    print(e.json())

# Example with invalid websocket URL (e.g., wrong scheme)
try:
    config_invalid = EndpointsConfig(
        service_url="https://api.example.com/resource",
        realtime_feed="ftp://feed.example.com/socket"  # invalid scheme for WebsocketUrl
    )
except ValidationError as e:
    print("\nValidation error for invalid EndpointsConfig:")
    print(e.json())
```

---

### Service Connection DSNs

This model demonstrates:

- **`PostgresDsn`** for PostgreSQL connection strings.
- **`AmqpDsn`** for AMQP connection strings (commonly used with RabbitMQ).

```python
from pydantic import BaseModel, PostgresDsn, AmqpDsn, ValidationError

class ServiceConnections(BaseModel):
    postgres_dsn: PostgresDsn
    amqp_dsn: AmqpDsn

# Correct usage
try:
    connections = ServiceConnections(
        postgres_dsn="postgresql://user:password@localhost:5432/mydatabase",
        amqp_dsn="amqp://guest:guest@localhost:5672/"
    )
    print("\nService Connections (Valid):")
    print(connections)
except ValidationError as e:
    print("Validation error in ServiceConnections:")
    print(e.json())

# Example with an invalid DSN (e.g., missing required parts)
try:
    connections_invalid = ServiceConnections(
        postgres_dsn="not a dsn",  # invalid DSN format
        amqp_dsn="amqp://guest:guest@localhost:5672/"
    )
except ValidationError as e:
    print("\nValidation error for invalid ServiceConnections:")
    print(e.json())
```

---

### Network Settings

This model uses:

- **`IPvAnyAddress`** for validating an IP address (IPv4 or IPv6).
- **`IPvAnyNetwork`** for validating an IP network (e.g., `192.168.1.0/24`).

```python
from pydantic import BaseModel, IPvAnyAddress, IPvAnyNetwork, ValidationError

class NetworkSettings(BaseModel):
    host_ip: IPvAnyAddress
    allowed_subnet: IPvAnyNetwork

# Correct usage
try:
    network = NetworkSettings(
        host_ip="192.168.1.42",
        allowed_subnet="192.168.1.0/24"
    )
    print("\nNetwork Settings (Valid):")
    print(network)
except ValidationError as e:
    print("Validation error in NetworkSettings:")
    print(e.json())

# Example with invalid IP address and subnet
try:
    network_invalid = NetworkSettings(
        host_ip="invalid-ip",
        allowed_subnet="192.168.1.0/33"  # invalid subnet mask
    )
except ValidationError as e:
    print("\nValidation error for invalid NetworkSettings:")
    print(e.json())
```

---

## Conclusion

Pydantic offers a robust and intuitive way to define, validate, and parse data using Python’s type hints. In this article, we explored various Pydantic types—from handling sensitive data and payment card numbers to validating file paths, URLs, connection strings, and network settings. By categorizing the examples into **Sensitive Data and Payment Models**, **File System and Encoding Models**, and **Network and Connection Models**, you can quickly locate the type of validation that fits your needs.

Each example includes error handling to help you understand what happens when validations fail, ensuring that your applications are both robust and easy to maintain. Happy coding!

---