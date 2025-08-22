import os
import streamlit as st
import pymongo
import bcrypt
from datetime import datetime, timedelta
import uuid
from bson.objectid import ObjectId
import gridfs
import bson

from utils import get_quantum_flag, quantum_safe_encrypt_b64, quantum_safe_decrypt_b64

class MongoDB:
    def __init__(self, connection_string=None):
        """Initialize MongoDB connection."""
        self.connection_string = connection_string or os.environ.get(
            "MONGODB_URI", "mongodb://localhost:27017/"
        )
        self.client = None
        self.db = None
        self.fs = None
        self.connect()

    def connect(self):
        """Establish connection to MongoDB."""
        try:
            self.client = pymongo.MongoClient(self.connection_string)
            self.db = self.client.rag_system
            self.fs = gridfs.GridFS(self.db)
            self.client.admin.command('ping')
            st.sidebar.success("Connected to MongoDB")
        except Exception as e:
            st.sidebar.error(f"MongoDB Connection Error: {str(e)}")
            self.client = None
            self.db = None
            self.fs = None

    # ------------------------
    # Users and sessions
    # ------------------------
    def create_user(self, email, password, name):
        """Create a new user with hashed password."""
        if self.client is None:
            return False, "Database connection not established"
        if self.db.users.find_one({"email": email}):
            return False, "User with this email already exists"

        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
        user = {
            "email": email,
            "password": hashed_password,
            "name": name,
            "created_at": datetime.now(),
            "last_login": None,
            "documents": [],
            "notebooks": [],
            "usage_stats": {
                "total_docs": 0,
                "total_pdfs": 0,
                "total_queries": 0,
                "rag_documents": 0,
                "last_activity": datetime.now()
            }
        }
        try:
            result = self.db.users.insert_one(user)
            return True, str(result.inserted_id)
        except Exception as e:
            return False, str(e)

    def authenticate_user(self, email, password):
        """Authenticate user with email and password."""
        if self.client is None:
            return False, "Database connection not established"

        user = self.db.users.find_one({"email": email})
        if not user:
            return False, "Invalid email or password"

        if bcrypt.checkpw(password.encode('utf-8'), user['password']):
            self.db.users.update_one(
                {"_id": user["_id"]},
                {"$set": {"last_login": datetime.now()}}
            )

            session_id = str(uuid.uuid4())
            expiry = datetime.now() + timedelta(days=1)
            self.db.sessions.insert_one({
                "session_id": session_id,
                "user_id": user["_id"],
                "expiry": expiry
            })
            return True, {
                "user_id": str(user["_id"]),
                "name": user["name"],
                "email": user["email"],
                "session_id": session_id
            }
        else:
            return False, "Invalid email or password"

    def validate_session(self, session_id):
        """Validate an existing session."""
        if self.client is None:
            return False, "Database connection not established"

        session = self.db.sessions.find_one({
            "session_id": session_id,
            "expiry": {"$gt": datetime.now()}
        })
        if not session:
            return False, "Session expired or invalid"

        user = self.db.users.find_one({"_id": session["user_id"]})
        if not user:
            return False, "User not found"

        return True, {
            "user_id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"]
        }

    def logout_user(self, session_id):
        """Invalidate a user session for logout."""
        if self.client is None:
            return False, "Database connection not established"
        try:
            self.db.sessions.delete_one({"session_id": session_id})
            return True, "Logged out successfully"
        except Exception as e:
            return False, str(e)

    # ------------------------
    # Files (GridFS)
    # ------------------------
    def save_document_file(self, file_data, filename, file_type, user_id, notebook_id=None, custom_name=None):
        """Save a document file to GridFS (with optional quantum-safe encryption)."""
        if self.client is None or self.fs is None:
            st.error("Database connection not established")
            return False, "Database connection not established"

        try:
            display_name = custom_name if custom_name else filename
            metadata = {
                "filename": filename,
                "display_name": display_name,
                "file_type": file_type,
                "user_id": ObjectId(user_id),
                "notebook_id": notebook_id,
                "upload_date": datetime.now()
            }

            # Normalize file_data to bytes
            if not isinstance(file_data, bytes):
                if hasattr(file_data, 'read'):
                    file_data = file_data.read()
                else:
                    file_data = bytes(file_data)

            payload = file_data
            if get_quantum_flag("enable_quantum_secure_storage"):
                # Important: in production, use a KMS-managed secret instead of a hardcoded passphrase.
                payload_b64 = quantum_safe_encrypt_b64(file_data, passphrase="embedding_store_kms")
                payload = payload_b64.encode("utf-8")
                metadata["qs_encrypted"] = True

            st.sidebar.info(f"Saving file: {filename} ({len(payload)} bytes)")
            file_id = self.fs.put(payload, **metadata)

            # Usage counters
            self.db.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$inc": {"usage_stats.total_docs": 1}}
            )
            if file_type == "pdf":
                self.db.users.update_one(
                    {"_id": ObjectId(user_id)},
                    {"$inc": {"usage_stats.total_pdfs": 1}}
                )
            if notebook_id:
                self.db.notebooks.update_one(
                    {"_id": ObjectId(notebook_id)},
                    {"$inc": {"document_count": 1}}
                )
            return True, str(file_id)
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")
            return False, str(e)

    def get_document_file(self, file_id):
        """Retrieve a document file from GridFS (with optional quantum-safe decryption)."""
        if self.client is None or self.fs is None:
            return False, "Database connection not established"
        try:
            file_obj = self.fs.get(ObjectId(file_id))
            raw = file_obj.read()

            # GridFS wraps metadata differently; try both attribute and internal dict
            is_qs = getattr(file_obj, "qs_encrypted", False)
            if not is_qs:
                try:
                    is_qs = file_obj._file.get("qs_encrypted", False)  # type: ignore[attr-defined]
                except Exception:
                    is_qs = False

            data = raw
            if is_qs and get_quantum_flag("enable_quantum_secure_storage"):
                # Stored as base64 string if encrypted
                try:
                    data = quantum_safe_decrypt_b64(raw.decode("utf-8"), passphrase="embedding_store_kms") or raw
                except Exception:
                    data = raw

            return True, {
                "data": data,
                "filename": file_obj.filename,
                "display_name": getattr(file_obj, "display_name", file_obj.filename),
                "file_type": getattr(file_obj, "file_type", "unknown"),
                "upload_date": getattr(file_obj, "upload_date", datetime.now())
            }
        except Exception as e:
            return False, str(e)

    def list_user_documents(self, user_id, notebook_id=None):
        """List all documents for a user, optionally filtered by notebook."""
        if self.client is None or self.fs is None:
            return False, "Database connection not established"
        try:
            query = {"user_id": ObjectId(user_id)}
            if notebook_id:
                query["notebook_id"] = notebook_id
            cursor = self.db.fs.files.find(query).sort("uploadDate", pymongo.DESCENDING)
            result = []
            for file_doc in cursor:
                result.append({
                    "file_id": str(file_doc["_id"]),
                    "filename": file_doc["filename"],
                    "display_name": file_doc.get("display_name", file_doc["filename"]),
                    "file_type": file_doc.get("file_type", "unknown"),
                    "upload_date": file_doc.get("upload_date", file_doc.get("uploadDate", datetime.now())),
                    "notebook_id": file_doc.get("notebook_id")
                })
            return True, result
        except Exception as e:
            st.error(f"Error listing documents: {str(e)}")
            return False, str(e)

    def delete_document(self, file_id, user_id):
        """Delete a document from GridFS."""
        if self.client is None or self.fs is None:
            return False, "Database connection not established"
        try:
            file_doc = self.db.fs.files.find_one({"_id": ObjectId(file_id)})
            if not file_doc:
                return False, "Document not found"
            if str(file_doc.get("user_id", "")) != user_id:
                return False, "You don't have permission to delete this document"

            notebook_id = file_doc.get("notebook_id")
            file_type = file_doc.get("file_type", "unknown")

            self.fs.delete(ObjectId(file_id))

            self.db.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$inc": {"usage_stats.total_docs": -1}}
            )
            if file_type == "pdf":
                self.db.users.update_one(
                    {"_id": ObjectId(user_id)},
                    {"$inc": {"usage_stats.total_pdfs": -1}}
                )
            if notebook_id:
                self.db.notebooks.update_one(
                    {"_id": ObjectId(notebook_id)},
                    {"$inc": {"document_count": -1}}
                )
            return True, "Document deleted successfully"
        except Exception as e:
            st.error(f"Error deleting document: {str(e)}")
            return False, str(e)

    # ------------------------
    # Notebooks
    # ------------------------
    def create_notebook(self, user_id, name, description="", color="#1E88E5", metadata=None):
        """Create a new notebook."""
        try:
            import datetime as dt
            now = dt.datetime.now()
            notebook = {
                "name": name,
                "description": description,
                "color": color,
                "user_id": ObjectId(user_id),
                "created_at": now,
                "last_accessed": now,
                "is_favorite": False,
                "domains": (metadata.get("domains", ["General"]) if metadata else ["General"]),
                "metadata": metadata or {}
            }
            result = self.db.notebooks.insert_one(notebook)
            return True, result.inserted_id
        except Exception as e:
            return False, f"Error creating notebook: {str(e)}"

    def get_notebooks(self, user_id):
        """Get all notebooks for a user."""
        if self.client is None:
            return False, "Database connection not established"
        try:
            notebooks = list(self.db.notebooks.find({"user_id": ObjectId(user_id)}).sort("last_accessed", -1))
            for notebook in notebooks:
                notebook["_id"] = str(notebook["_id"])
                notebook["user_id"] = str(notebook["user_id"])
            return True, notebooks
        except Exception as e:
            return False, str(e)

    def get_notebook(self, notebook_id):
        """Get a specific notebook by ID."""
        if self.client is None:
            return False, "Database connection not established"
        try:
            notebook = self.db.notebooks.find_one({"_id": ObjectId(notebook_id)})
            if not notebook:
                return False, "Notebook not found"
            self.db.notebooks.update_one(
                {"_id": ObjectId(notebook_id)},
                {"$set": {"last_accessed": datetime.now()}}
            )
            notebook["_id"] = str(notebook["_id"])
            notebook["user_id"] = str(notebook["user_id"])
            return True, notebook
        except Exception as e:
            return False, str(e)

    def update_notebook(self, notebook_id, data):
        """Update notebook properties."""
        if self.client is None:
            return False, "Database connection not established"
        try:
            if "_id" in data:
                del data["_id"]
            if "user_id" in data:
                del data["user_id"]
            result = self.db.notebooks.update_one(
                {"_id": ObjectId(notebook_id)},
                {"$set": data}
            )
            if result.matched_count == 0:
                return False, "Notebook not found"
            return True, "Notebook updated successfully"
        except Exception as e:
            return False, str(e)

    def toggle_favorite_notebook(self, notebook_id):
        """Toggle favorite status for a notebook."""
        if self.client is None:
            return False, "Database connection not established"
        try:
            notebook = self.db.notebooks.find_one({"_id": ObjectId(notebook_id)})
            if not notebook:
                return False, "Notebook not found"
            new_status = not notebook.get("is_favorite", False)
            self.db.notebooks.update_one(
                {"_id": ObjectId(notebook_id)},
                {"$set": {"is_favorite": new_status}}
            )
            return True, new_status
        except Exception as e:
            return False, str(e)

    def delete_notebook(self, notebook_id, user_id):
        """Delete a notebook and remove reference from user."""
        if self.client is None:
            return False, "Database connection not established"
        try:
            result = self.db.notebooks.delete_one({
                "_id": ObjectId(notebook_id),
                "user_id": ObjectId(user_id)
            })
            if result.deleted_count == 0:
                return False, "Notebook not found or not owned by user"

            self.db.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$pull": {"notebooks": ObjectId(notebook_id)}}
            )

            self.db.fs.files.update_many(
                {"notebook_id": notebook_id},
                {"$set": {"notebook_id": None}}
            )
            return True, "Notebook deleted successfully"
        except Exception as e:
            return False, str(e)

    # ------------------------
    # RAG metadata and analytics
    # ------------------------
    def save_document_metadata(self, user_id, document_info, notebook_id=None):
        """Save metadata about processed documents."""
        if self.client is None:
            return False, "Database connection not established"
        try:
            document_info["processed_at"] = datetime.now()
            if notebook_id:
                document_info["notebook_id"] = notebook_id

            if "documents" in document_info and notebook_id:
                document_count = len(document_info.get("documents", []))
                self.db.notebooks.update_one(
                    {"_id": ObjectId(notebook_id)},
                    {"$inc": {"rag_document_count": document_count}}
                )

            self.db.rag_documents.insert_one({
                "user_id": ObjectId(user_id),
                "notebook_id": notebook_id,
                "metadata": document_info,
                "created_at": datetime.now()
            })

            self.db.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$inc": {"usage_stats.rag_documents": len(document_info.get("documents", []))}}
            )
            return True, "Document metadata saved"
        except Exception as e:
            return False, str(e)

    def get_rag_document_history(self, user_id, notebook_id=None):
        """Get RAG document processing history for a user."""
        if self.client is None:
            return False, "Database connection not established"
        try:
            query = {"user_id": ObjectId(user_id)}
            if notebook_id:
                query["notebook_id"] = notebook_id
            documents = list(self.db.rag_documents.find(query).sort("created_at", -1))
            for doc in documents:
                doc["_id"] = str(doc["_id"])
                doc["user_id"] = str(doc["user_id"])
                if doc.get("notebook_id"):
                    doc["notebook_id"] = str(doc["notebook_id"])
            return True, documents
        except Exception as e:
            return False, str(e)

    def log_query(self, user_id, query, response_time, notebook_id=None):
        """Log a query for analytics."""
        if self.client is None:
            return False, "Database connection not established"
        try:
            query_log = {
                "user_id": ObjectId(user_id),
                "notebook_id": notebook_id,
                "query": query,
                "response_time": response_time,
                "timestamp": datetime.now()
            }
            self.db.query_logs.insert_one(query_log)
            self.db.users.update_one(
                {"_id": ObjectId(user_id)},
                {
                    "$inc": {"usage_stats.total_queries": 1},
                    "$set": {"usage_stats.last_activity": datetime.now()}
                }
            )
            return True, "Query logged"
        except Exception as e:
            return False, str(e)

    def get_user_analytics(self, user_id):
        """Get analytics for a user."""
        if self.client is None:
            return False, "Database connection not established"
        try:
            user = self.db.users.find_one({"_id": ObjectId(user_id)})
            if not user:
                return False, "User not found"

            usage_stats = user.get("usage_stats", {})
            query_logs = list(self.db.query_logs.find({"user_id": ObjectId(user_id)}).sort("timestamp", -1).limit(100))
            if query_logs:
                avg_response_time = sum(log.get("response_time", 0) for log in query_logs) / len(query_logs)
            else:
                avg_response_time = 0

            notebooks = list(self.db.notebooks.find({"user_id": ObjectId(user_id)}))
            notebook_stats = [{
                "id": str(notebook["_id"]),
                "name": notebook["name"],
                "document_count": notebook.get("document_count", 0),
                "rag_document_count": notebook.get("rag_document_count", 0),
                "created_at": notebook["created_at"],
                "last_accessed": notebook.get("last_accessed", notebook["created_at"])
            } for notebook in notebooks]

            analytics = {
                "user_id": str(user["_id"]),
                "name": user["name"],
                "email": user["email"],
                "created_at": user["created_at"],
                "last_login": user.get("last_login"),
                "total_documents": usage_stats.get("total_docs", 0),
                "total_pdfs": usage_stats.get("total_pdfs", 0),
                "total_queries": usage_stats.get("total_queries", 0),
                "total_rag_documents": usage_stats.get("rag_documents", 0),
                "last_activity": usage_stats.get("last_activity"),
                "avg_response_time": avg_response_time,
                "notebook_count": len(notebooks),
                "notebook_stats": notebook_stats,
                "recent_queries": [{
                    "query": log["query"],
                    "timestamp": log["timestamp"],
                    "response_time": log.get("response_time", 0),
                    "notebook_id": log.get("notebook_id")
                } for log in query_logs[:10]]
            }
            return True, analytics
        except Exception as e:
            return False, str(e)

    def get_notebook_analytics(self, notebook_id):
        """Get analytics for a specific notebook."""
        if self.client is None:
            return False, "Database connection not established"
        try:
            notebook = self.db.notebooks.find_one({"_id": ObjectId(notebook_id)})
            if not notebook:
                return False, "Notebook not found"

            document_count = notebook.get("document_count", 0)
            rag_document_count = notebook.get("rag_document_count", 0)
            query_logs = list(self.db.query_logs.find({"notebook_id": notebook_id}).sort("timestamp", -1))
            if query_logs:
                avg_response_time = sum(log.get("response_time", 0) for log in query_logs) / len(query_logs)
            else:
                avg_response_time = 0

            analytics = {
                "notebook_id": str(notebook["_id"]),
                "name": notebook["name"],
                "description": notebook.get("description", ""),
                "created_at": notebook["created_at"],
                "last_accessed": notebook.get("last_accessed", notebook["created_at"]),
                "document_count": document_count,
                "rag_document_count": rag_document_count,
                "query_count": len(query_logs),
                "avg_response_time": avg_response_time,
                "recent_queries": [{
                    "query": log["query"],
                    "timestamp": log["timestamp"],
                    "response_time": log.get("response_time", 0)
                } for log in query_logs[:10]]
            }
            return True, analytics
        except Exception as e:
            return False, str(e)

    # ------------------------
    # FAISS index storage (vectors)
    # ------------------------
    def save_faiss_index(self, notebook_id, user_id, index_binary, documents_binary, metadata=None):
        """Save FAISS index data for a notebook (with optional quantum-safe encryption)."""
        try:
            import datetime as dt
            collection = self.db.faiss_indexes
            existing = collection.find_one({"notebook_id": notebook_id})
            now = dt.datetime.now()

            record_meta = metadata or {}
            payload_index = index_binary
            payload_docs = documents_binary

            if get_quantum_flag("enable_quantum_secure_storage"):
                # Store as base64 strings for easy transport; encrypted at rest
                payload_index_b64 = quantum_safe_encrypt_b64(index_binary, passphrase="faiss_store_kms").encode("utf-8")
                record_meta["qs_encrypted_index"] = True
                payload_index = payload_index_b64

                if documents_binary and len(documents_binary) > 0:
                    payload_docs_b64 = quantum_safe_encrypt_b64(documents_binary, passphrase="faiss_store_kms").encode("utf-8")
                    record_meta["qs_encrypted_docs"] = True
                    payload_docs = payload_docs_b64

            data = {
                "notebook_id": notebook_id,
                "user_id": user_id,
                "faiss_index": bson.Binary(payload_index),
                "has_documents": bool(payload_docs) and len(payload_docs) > 0,
                "metadata": record_meta,
                "updated_at": now
            }
            if payload_docs and len(payload_docs) > 0:
                data["documents"] = bson.Binary(payload_docs)

            if existing:
                collection.update_one({"notebook_id": notebook_id}, {"$set": data})
                return True, "FAISS index updated successfully"
            else:
                data["created_at"] = now
                collection.insert_one(data)
                return True, "FAISS index created successfully"
        except Exception as e:
            print(f"MongoDB error saving FAISS index: {str(e)}")
            return False, f"Error saving FAISS index: {str(e)}"

    def get_faiss_index(self, notebook_id):
        """Retrieve FAISS index data for a notebook (with optional quantum-safe decryption)."""
        try:
            collection = self.db.faiss_indexes
            result = collection.find_one({"notebook_id": notebook_id})
            if not result:
                return False, "No FAISS index found for this notebook"

            meta = result.get("metadata", {})
            index_blob = bytes(result["faiss_index"])
            docs_blob = bytes(result.get("documents", b""))

            if get_quantum_flag("enable_quantum_secure_storage") and meta.get("qs_encrypted_index"):
                try:
                    dec = quantum_safe_decrypt_b64(index_blob.decode("utf-8"), passphrase="faiss_store_kms")
                    if dec is not None:
                        index_blob = dec
                except Exception:
                    pass

            if get_quantum_flag("enable_quantum_secure_storage") and meta.get("qs_encrypted_docs") and docs_blob:
                try:
                    decd = quantum_safe_decrypt_b64(docs_blob.decode("utf-8"), passphrase="faiss_store_kms")
                    if decd is not None:
                        docs_blob = decd
                except Exception:
                    pass

            return True, {
                "faiss_index": index_blob,
                "documents": docs_blob,
                "metadata": meta,
                "updated_at": result["updated_at"]
            }
        except Exception as e:
            print(f"MongoDB error getting FAISS index: {str(e)}")
            return False, f"Error retrieving FAISS index: {str(e)}"