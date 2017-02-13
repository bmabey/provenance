from datetime import datetime
from memoized_property import memoized_property
from sqlalchemy import Column, Integer, ForeignKey, Table
from sqlalchemy.orm import deferred, relationship
from sqlalchemy.dialects.postgresql import BOOLEAN, BYTEA, FLOAT, INTEGER, JSONB, TIMESTAMP, VARCHAR
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

SHA1_LENGTH = 40
VALUE_ID_LENGTH = SHA1_LENGTH + 10 # extra 10 for optional file extension info

class Artifact(Base):
    __tablename__ = 'artifacts'

    id = Column(VARCHAR(SHA1_LENGTH), primary_key=True)
    value_id = Column(VARCHAR(VALUE_ID_LENGTH))

    name = Column(VARCHAR(1000))
    version = Column(INTEGER)
    fn_module = Column(VARCHAR(100))
    fn_name = Column(VARCHAR(100))

    composite = Column(BOOLEAN)

    value_id_duration = Column(FLOAT)
    compute_duration = Column(FLOAT)
    hash_duration = Column(FLOAT)

    computed_at = Column(TIMESTAMP)
    added_at = Column(TIMESTAMP, default=datetime.utcnow)

    host = Column(JSONB)
    process = Column(JSONB)
    expanded_inputs = deferred(Column(JSONB))
    serializer = Column(VARCHAR(128), default='joblib')
    load_kwargs = Column(JSONB)
    dump_kwargs = Column(JSONB)
    custom_fields = Column(JSONB)

    def __init__(self, artifact, expanded_inputs):
        self.id = artifact.id
        self.value_id = artifact.value_id
        self.name = artifact.name
        self.version = artifact.version
        self.fn_module = artifact.fn_module
        self.fn_name = artifact.fn_name
        self.composite = artifact.composite
        self.value_id_duration = artifact.value_id_duration
        self.compute_duration = artifact.compute_duration
        self.hash_duration = artifact.hash_duration
        self.host = artifact.host
        self.process = artifact.process
        self.expanded_inputs = expanded_inputs
        self.custom_fields = artifact.custom_fields
        self.computed_at = artifact.computed_at
        self.serializer = artifact.serializer
        self.load_kwargs = artifact.load_kwargs
        self.dump_kwargs = artifact.dump_kwargs

    @memoized_property
    def props(self):
        return {'id': self.id,
                'value_id': self.value_id,
                'name': self.name,
                'version': self.version,
                'fn_module': self.fn_module,
                'fn_name': self.fn_name,
                'composite': self.composite,
                'value_id_duration': self.value_id_duration,
                'compute_duration': self.compute_duration,
                'hash_duration': self.hash_duration,
                'host': self.host,
                'process': self.process,
                'serializer': self.serializer,
                'load_kwargs': self.load_kwargs,
                'dump_kwargs': self.dump_kwargs,
                'custom_fields': self.custom_fields,
                'computed_at': self.computed_at}

    def __repr__(self):
        return '<Artifact %r>' % self.id


class ArtifactSet(Base):
    __tablename__ = 'artifact_sets'

    id = Column(Integer, primary_key=True)
    set_id = Column(VARCHAR(SHA1_LENGTH))
    name = Column(VARCHAR(1000))
    created_at = Column(TIMESTAMP, default=datetime.utcnow)


    def __init__(self, artifact_set):
        self.set_id = artifact_set.id
        self.name = artifact_set.name
        self.created_at = artifact_set.created_at

    @memoized_property
    def props(self):
        return {'id': self.set_id,
                'name': self.name,
                'created_at': self.created_at}

    def __repr__(self):
        return '<ArtifactSet %r, %r>' % (self.set_id, self.name)


class ArtifactSetMember(Base):
    __tablename__ = 'artifact_set_members'

    set_id = Column(VARCHAR(SHA1_LENGTH), #ForeignKey("artifact_sets.set_id"),
                    primary_key=True)
    artifact_id = Column(VARCHAR(SHA1_LENGTH), #ForeignKey("artifacts.id"),
                         primary_key=True)
