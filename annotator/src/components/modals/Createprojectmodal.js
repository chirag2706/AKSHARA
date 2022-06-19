const Createprojectmodal = (props) => {
    let creatingProject = props.creatingProject;
    let handleCreateProject = props.handleCreateProject;

    return (
        <div className="modal fade" id="CreateProjectModal" tabIndex="-1" aria-labelledby="CreateProjectModalLabel" aria-hidden="true">
            <div className="modal-dialog">
                <div className="modal-content">
                    <div className="modal-header">
                        <h5 className="modal-title">Create New Project</h5>
                        <button type="button" id="createProjectModalClose" className="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div className="modal-body">
                        <form>
                            <div className="mb-3">
                                <label htmlFor="createModalTitleInput" className="col-form-label" aria-required="true">
                                    Title:
                                </label>
                                <input type="text" className="form-control" id="createModalTitleInput" maxLength={32} placeholder="max 32 characters" />
                                <div id="createModalTitleHelperText" style={{ fontSize: "10px", color: "red", paddingLeft: "5px", display: "none" }}>
                                    field is required
                                </div>
                            </div>
                            <div className="mb-3">
                                <label htmlFor="createModalDescriptionInput" className="col-form-label">
                                    Description:
                                </label>
                                <textarea className="form-control" id="createModalDescriptionInput"></textarea>
                                <div id="createModalDescriptionHelperText" style={{ fontSize: "10px", color: "red", paddingLeft: "5px", display: "none" }}>
                                    field is required
                                </div>
                            </div>
                        </form>
                        {creatingProject ? (
                            <div>
                                <div className="d-flex justify-content-center">
                                    <div className="spinner-border" role="status" style={{ color: "#4285F4" }}></div>
                                </div>
                                <div style={{ textAlign: "center", paddingTop: "20px" }}>Creating projects. Please wait</div>
                            </div>
                        ) : (
                            <></>
                        )}
                    </div>
                    <div className="modal-footer">
                        <button type="button" className="btn btn-secondary" data-bs-dismiss="modal">
                            Close
                        </button>
                        <button type="button" className="btn btn-primary" onClick={() => {handleCreateProject()}}>
                            Create Project
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Createprojectmodal;
