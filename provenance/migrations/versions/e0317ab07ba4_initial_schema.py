"""initial schema

Revision ID: e0317ab07ba4
Revises: 
Create Date: 2017-03-13 13:33:59.644604

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import sqlalchemy.dialects.postgresql as pg

# revision identifiers, used by Alembic.
revision = 'e0317ab07ba4'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('artifact_set_members',
                    sa.Column('set_id', sa.VARCHAR(length=40), nullable=False),
                    sa.Column('artifact_id', sa.VARCHAR(length=40), nullable=False),
                    sa.PrimaryKeyConstraint('set_id', 'artifact_id'))

    op.create_table('artifact_sets',
                    sa.Column('id', sa.INTEGER(), nullable=False),
                    sa.Column('set_id', sa.VARCHAR(length=40), nullable=True),
                    sa.Column('name', sa.VARCHAR(length=1000), nullable=True),
                    sa.Column('created_at', pg.TIMESTAMP(), nullable=True),
                    sa.PrimaryKeyConstraint('id'))

    op.create_table('runs',
                    sa.Column('id', sa.VARCHAR(length=40), nullable=False),
                    sa.Column('hostname', sa.VARCHAR(length=256), nullable=True),
                    sa.Column('info', pg.JSONB(), nullable=True),
                    sa.Column('created_at', pg.TIMESTAMP(), nullable=True),
                    sa.PrimaryKeyConstraint('id'))

    op.create_table('artifacts',
                    sa.Column('id', sa.VARCHAR(length=40), nullable=False),
                    sa.Column('value_id', sa.VARCHAR(length=50), nullable=True),
                    sa.Column('run_id', sa.VARCHAR(length=40), nullable=True),
                    sa.Column('name', sa.VARCHAR(length=1000), nullable=True),
                    sa.Column('version', sa.INTEGER(), nullable=True),
                    sa.Column('fn_module', sa.VARCHAR(length=100), nullable=True),
                    sa.Column('fn_name', sa.VARCHAR(length=100), nullable=True),
                    sa.Column('composite', sa.BOOLEAN(), nullable=True),
                    sa.Column('value_id_duration', sa.FLOAT(), nullable=True),
                    sa.Column('compute_duration', sa.FLOAT(), nullable=True),
                    sa.Column('hash_duration', sa.FLOAT(), nullable=True),
                    sa.Column('computed_at', pg.TIMESTAMP(), nullable=True),
                    sa.Column('added_at', pg.TIMESTAMP(), nullable=True),
                    sa.Column('input_artifact_ids', pg.ARRAY(pg.VARCHAR(length=40)), nullable=True),
                    sa.Column('inputs_json', pg.JSONB(), nullable=True),
                    sa.Column('serializer', sa.VARCHAR(length=128), nullable=True),
                    sa.Column('load_kwargs', pg.JSONB(), nullable=True),
                    sa.Column('dump_kwargs', pg.JSONB(), nullable=True),
                    sa.Column('custom_fields', pg.JSONB(), nullable=True),
                    sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ),
                    sa.PrimaryKeyConstraint('id'))


def downgrade():
    op.drop_table('artifacts')
    op.drop_table('runs')
    op.drop_table('artifact_sets')
    op.drop_table('artifact_set_members')
