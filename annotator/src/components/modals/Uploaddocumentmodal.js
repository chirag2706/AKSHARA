const Uploaddocumentmodal = (props) => {
    let uploadingDocument = props.uploadingDocument;
    let handleUploadDocument = props.handleUploadDocument;

    return (
        <div className="modal fade" id="UploadDocumentModal" tabIndex="-1" aria-labelledby="UploadDocumentModalLabel" aria-hidden="true">
            <div className="modal-dialog">
                <div className="modal-content">
                    <div className="modal-header">
                        <h5 className="modal-title">Upload Document</h5>
                        <button id="uploadDocumentClose" type="button" className="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div className="modal-body">
                        <form>
                            <div className="mb-3">
                                <label htmlFor="uploadDocumentTitleInput" className="col-form-label" aria-required="true">
                                    Title:
                                </label>
                                <input type="text" className="form-control" id="uploadDocumentTitleInput" maxLength={32} placeholder="max 32 characters" />
                                <div id="uploadDocumentTitleHelperText" style={{ fontSize: "10px", color: "red", paddingLeft: "5px", display: "none" }}>
                                    field is required
                                </div>
                            </div>
                            <div className="mb-3">
                                <label htmlFor="uploadDocumentDescriptionInput" className="col-form-label">
                                    Description:
                                </label>
                                <textarea className="form-control" id="uploadDocumentDescriptionInput"></textarea>
                                <div id="uploadDocumentDescriptionHelperText" style={{ fontSize: "10px", color: "red", paddingLeft: "5px", display: "none" }}>
                                    field is required
                                </div>
                            </div>
                            <div className="mb-3">
                                <label htmlFor="uploadDocumentImageInput" className="col-form-label">
                                    Image:
                                </label>
                                <input type='file' accept="image/png, image/jpeg, image/jpg" className="form-control" id="uploadDocumentImageInput" />
                                <div id="uploadDocumentImageHelperText" style={{ fontSize: "10px", color: "red", paddingLeft: "5px", display: "none" }}>
                                    field is required
                                </div>
                            </div>
                        </form>
                        {uploadingDocument ? (
                            <div>
                                <div className="d-flex justify-content-center">
                                    <div className="spinner-border" role="status" style={{ color: "#4285F4" }}></div>
                                </div>
                                <div style={{ textAlign: "center", paddingTop: "20px" }}>Uploading document. Please wait</div>
                            </div>
                        ) : (
                            <></>
                        )}
                    </div>
                    <div className="modal-footer">
                        <button type="button" className="btn btn-secondary" data-bs-dismiss="modal">
                            Close
                        </button>
                        <button type="button" className="btn btn-primary" onClick={() => {handleUploadDocument()}}>
                            Upload
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Uploaddocumentmodal;
